//! Structured request and response types for `POST /v1/decisions`.

use crate::protocol::ApiError;
use eider_runtime::decision::{
    DecisionAnswer, DecisionCompletion, DecisionPromptQuestion, DecisionPromptRequest,
    MAX_DECISION_ANSWERS, MAX_DECISION_QUESTIONS,
};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Strict non-streaming decision request.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionApiRequest {
    pub model: String,
    pub state: Value,
    pub questions: IndexMap<String, DecisionQuestion>,
}

/// One typed decision question.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum DecisionQuestion {
    Noul {
        instructions: Value,
        #[serde(default)]
        criteria: NoulCriteria,
    },
    Choice {
        instructions: Value,
        criteria: IndexMap<String, Option<String>>,
    },
    Score {
        instructions: Value,
        criteria: Vec<String>,
    },
}

/// Optional meanings for the binary answers.
#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NoulCriteria {
    #[serde(rename = "true")]
    #[serde(default = "default_true_description")]
    pub true_description: String,
    #[serde(rename = "false")]
    #[serde(default = "default_false_description")]
    pub false_description: String,
}

impl Default for NoulCriteria {
    fn default() -> Self {
        Self {
            true_description: default_true_description(),
            false_description: default_false_description(),
        }
    }
}

fn default_true_description() -> String {
    "Yes".to_string()
}

fn default_false_description() -> String {
    "No".to_string()
}

impl DecisionApiRequest {
    /// Validates the public schema and converts it to the prompt compiler input.
    pub fn into_prompt_request(self) -> Result<DecisionPromptRequest, ApiError> {
        if self.model.trim().is_empty() {
            return Err(ApiError::invalid("model", "model must not be empty"));
        }
        validate_content("state", &self.state)?;
        if self.questions.is_empty() || self.questions.len() > MAX_DECISION_QUESTIONS {
            return Err(ApiError::invalid(
                "questions",
                format!("questions must contain 1..={MAX_DECISION_QUESTIONS} entries"),
            ));
        }
        let mut questions = Vec::with_capacity(self.questions.len());
        for (id, question) in self.questions {
            if id.is_empty() {
                return Err(ApiError::invalid(
                    "questions",
                    "question IDs must not be empty",
                ));
            }
            let question = match question {
                DecisionQuestion::Noul {
                    instructions,
                    criteria,
                } => {
                    validate_content(&format!("questions.{id}.instructions"), &instructions)?;
                    DecisionPromptQuestion::Noul {
                        instructions,
                        true_description: criteria.true_description,
                        false_description: criteria.false_description,
                    }
                }
                DecisionQuestion::Choice {
                    instructions,
                    criteria,
                } => {
                    validate_content(&format!("questions.{id}.instructions"), &instructions)?;
                    validate_answer_count(&id, criteria.len())?;
                    if criteria.keys().any(|key| key.is_empty()) {
                        return Err(ApiError::invalid(
                            format!("questions.{id}.criteria"),
                            "choice option names must not be empty",
                        ));
                    }
                    DecisionPromptQuestion::Choice {
                        instructions,
                        options: criteria.into_iter().collect(),
                    }
                }
                DecisionQuestion::Score {
                    instructions,
                    criteria,
                } => {
                    validate_content(&format!("questions.{id}.instructions"), &instructions)?;
                    validate_answer_count(&id, criteria.len())?;
                    DecisionPromptQuestion::Score {
                        instructions,
                        levels: criteria,
                    }
                }
            };
            questions.push((id, question));
        }
        Ok(DecisionPromptRequest {
            state: self.state,
            questions,
        })
    }
}

fn validate_answer_count(id: &str, count: usize) -> Result<(), ApiError> {
    if !(2..=MAX_DECISION_ANSWERS).contains(&count) {
        return Err(ApiError::invalid(
            format!("questions.{id}.criteria"),
            format!("criteria must contain 2..={MAX_DECISION_ANSWERS} entries"),
        ));
    }
    Ok(())
}

fn validate_content(param: &str, value: &Value) -> Result<(), ApiError> {
    if matches!(value, Value::String(_) | Value::Array(_) | Value::Object(_)) {
        return Ok(());
    }
    Err(ApiError::invalid(
        param,
        "value must be a string, object, or array",
    ))
}

/// Complete non-streaming decision response.
#[derive(Clone, Debug, Serialize)]
pub struct DecisionApiResponse {
    pub model: String,
    pub answers: IndexMap<String, DecisionApiAnswer>,
    pub usage: DecisionApiUsage,
}

/// One typed decision answer.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum DecisionApiAnswer {
    Noul {
        noul: f32,
    },
    Choice {
        choice: String,
        probabilities: IndexMap<String, f32>,
        confidence: f32,
    },
    Score {
        score: f32,
        legend: IndexMap<String, String>,
        probabilities: IndexMap<String, f32>,
        confidence: f32,
    },
}

/// Logical decision token accounting.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct DecisionApiUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// Validated probability maps fitted for one exact deployment contract.
#[derive(Clone, Debug, Deserialize)]
pub struct DecisionCalibration {
    schema: String,
    validated_dataset: bool,
    model: String,
    prompt_format: String,
    deployment: CalibrationDeployment,
    dataset: String,
    dataset_version: String,
    dataset_sha256: String,
    maps: BTreeMap<String, CalibrationMap>,
}

#[derive(Clone, Debug, Deserialize)]
struct CalibrationDeployment {
    requested_model: String,
    returned_model: String,
}

#[derive(Clone, Debug, Deserialize)]
struct CalibrationMap {
    method: String,
    temperature: f32,
}

impl DecisionCalibration {
    /// Parses and validates one calibration artifact for an exact served model.
    pub fn from_json(bytes: &[u8], expected_model: &str) -> Result<Self, ApiError> {
        let calibration: Self = serde_json::from_slice(bytes).map_err(|error| {
            ApiError::invalid(
                "decision_calibration",
                format!("invalid calibration artifact: {error}"),
            )
        })?;
        calibration.validate(expected_model)?;
        Ok(calibration)
    }

    fn validate(&self, expected_model: &str) -> Result<(), ApiError> {
        if self.schema != "eider-decision-calibration-v1" {
            return Err(ApiError::invalid(
                "decision_calibration",
                format!("unsupported calibration schema {:?}", self.schema),
            ));
        }
        if !self.validated_dataset {
            return Err(ApiError::invalid(
                "decision_calibration",
                "calibration artifact does not identify a validated dataset",
            ));
        }
        if self.model != expected_model || self.deployment.returned_model != expected_model {
            return Err(ApiError::invalid(
                "decision_calibration",
                format!(
                    "calibration model {:?} does not match served model {expected_model:?}",
                    self.model
                ),
            ));
        }
        if self.prompt_format != eider_runtime::decision::DECISION_PROMPT_FORMAT {
            return Err(ApiError::invalid(
                "decision_calibration",
                format!("unsupported prompt format {:?}", self.prompt_format),
            ));
        }
        if self.deployment.requested_model.is_empty()
            || self.dataset.is_empty()
            || self.dataset_version.is_empty()
            || self.dataset_sha256.len() != 64
            || !self
                .dataset_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(ApiError::invalid(
                "decision_calibration",
                "calibration provenance is incomplete",
            ));
        }
        let expected = [
            ("noul", "temperature_nll"),
            ("choice", "temperature_nll"),
            ("score", "ordinal_temperature_rps"),
        ];
        for (question_type, method) in expected {
            let fitted = self.maps.get(question_type).ok_or_else(|| {
                ApiError::invalid(
                    "decision_calibration",
                    format!("calibration map for {question_type:?} is missing"),
                )
            })?;
            if fitted.method != method
                || !fitted.temperature.is_finite()
                || fitted.temperature <= 0.0
            {
                return Err(ApiError::invalid(
                    "decision_calibration",
                    format!("calibration map for {question_type:?} is invalid"),
                ));
            }
        }
        Ok(())
    }

    fn temperature(&self, question_type: &str) -> f32 {
        self.maps
            .get(question_type)
            .expect("validated calibration contains all question types")
            .temperature
    }
}

impl DecisionApiResponse {
    /// Converts one engine completion to the public response shape.
    pub fn from_completion(
        model: String,
        completion: DecisionCompletion,
        calibration: Option<&DecisionCalibration>,
    ) -> Result<Self, ApiError> {
        let mut answers = IndexMap::new();
        for (id, answer) in completion.answers {
            let answer = match answer {
                DecisionAnswer::Noul { probability } => {
                    let noul = calibration.map_or(probability, |calibration| {
                        calibrate_probabilities(
                            &[probability, 1.0 - probability],
                            calibration.temperature("noul"),
                        )[0]
                    });
                    DecisionApiAnswer::Noul { noul }
                }
                DecisionAnswer::Choice {
                    choice,
                    probabilities,
                    confidence,
                } => {
                    let (choice, probabilities, confidence) = match calibration {
                        Some(calibration) => {
                            let keys = probabilities
                                .iter()
                                .map(|(key, _)| key.clone())
                                .collect::<Vec<_>>();
                            let values = probabilities
                                .iter()
                                .map(|(_, value)| *value)
                                .collect::<Vec<_>>();
                            let values =
                                calibrate_probabilities(&values, calibration.temperature("choice"));
                            let selected = argmax(&values);
                            (
                                keys[selected].clone(),
                                keys.into_iter().zip(values.iter().copied()).collect(),
                                concentration(&values),
                            )
                        }
                        None => (choice, probabilities, confidence),
                    };
                    DecisionApiAnswer::Choice {
                        choice,
                        probabilities: probabilities.into_iter().collect(),
                        confidence,
                    }
                }
                DecisionAnswer::Score {
                    score,
                    legend,
                    probabilities,
                    confidence,
                } => {
                    let (score, probabilities, confidence) = match calibration {
                        Some(calibration) => {
                            let values = calibrate_probabilities(
                                &probabilities,
                                calibration.temperature("score"),
                            );
                            let score = values
                                .iter()
                                .enumerate()
                                .map(|(index, probability)| index as f32 * probability)
                                .sum();
                            let confidence = concentration(&values);
                            (score, values, confidence)
                        }
                        None => (score, probabilities, confidence),
                    };
                    DecisionApiAnswer::Score {
                        score,
                        legend: legend
                            .into_iter()
                            .enumerate()
                            .map(|(index, description)| (index.to_string(), description))
                            .collect(),
                        probabilities: probabilities
                            .into_iter()
                            .enumerate()
                            .map(|(index, probability)| (index.to_string(), probability))
                            .collect(),
                        confidence,
                    }
                }
            };
            if answers.insert(id.clone(), answer).is_some() {
                return Err(ApiError::server(format!(
                    "decision engine returned duplicate question ID {id:?}"
                )));
            }
        }
        Ok(Self {
            model,
            answers,
            usage: DecisionApiUsage {
                input_tokens: completion.usage.input_tokens,
                output_tokens: completion.usage.output_tokens,
            },
        })
    }
}

fn calibrate_probabilities(probabilities: &[f32], temperature: f32) -> Vec<f32> {
    let scaled = probabilities
        .iter()
        .map(|value| value.max(1e-12).ln() / temperature)
        .collect::<Vec<_>>();
    let maximum = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut weights = scaled
        .into_iter()
        .map(|value| (value - maximum).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f32>();
    for weight in &mut weights {
        *weight /= total;
    }
    weights
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(index, _)| index)
}

fn concentration(probabilities: &[f32]) -> f32 {
    let entropy = probabilities
        .iter()
        .filter(|value| **value > 0.0)
        .map(|value| -value * value.ln())
        .sum::<f32>();
    (1.0 - entropy / (probabilities.len() as f32).ln()).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eider_runtime::decision::DecisionUsage;
    use serde_json::json;

    #[test]
    fn accepts_all_question_shapes() {
        let request: DecisionApiRequest = serde_json::from_value(json!({
            "model": "decision-latest",
            "state": {"ticket": "duplicate charge"},
            "questions": {
                "refund": {"type": "noul", "instructions": ["refund", "requested"]},
                "team": {
                    "type": "choice",
                    "instructions": {"select": "team"},
                    "criteria": {"billing": null, "technical": "software problem"}
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent?",
                    "criteria": ["routine", "urgent"]
                }
            }
        }))
        .expect("valid request");
        let prepared = request.into_prompt_request().expect("valid prompt input");
        assert_eq!(prepared.questions.len(), 3);
    }

    #[test]
    fn noul_criteria_members_default_independently() {
        let request: DecisionApiRequest = serde_json::from_value(json!({
            "model": "decision-latest",
            "state": "ticket",
            "questions": {
                "partial": {
                    "type": "noul",
                    "instructions": "Is this urgent?",
                    "criteria": {"true": "Time-sensitive"}
                }
            }
        }))
        .expect("partial noul criteria");
        let prompt = request.into_prompt_request().expect("valid prompt input");
        let DecisionPromptQuestion::Noul {
            true_description,
            false_description,
            ..
        } = &prompt.questions[0].1
        else {
            panic!("question is noul");
        };
        assert_eq!(true_description, "Time-sensitive");
        assert_eq!(false_description, "No");
    }

    #[test]
    fn preserves_question_and_option_order() {
        let request: DecisionApiRequest = serde_json::from_str(
            r#"{
                "model":"decision-latest",
                "state":"ticket",
                "questions":{
                    "z_first":{"type":"choice","instructions":"Pick","criteria":{"zeta":null,"alpha":null}},
                    "a_second":{"type":"noul","instructions":"True?"}
                }
            }"#,
        )
        .expect("valid ordered request");
        let prompt = request.into_prompt_request().expect("valid prompt input");
        assert_eq!(prompt.questions[0].0, "z_first");
        assert_eq!(prompt.questions[1].0, "a_second");
        let DecisionPromptQuestion::Choice { options, .. } = &prompt.questions[0].1 else {
            panic!("first question is a choice");
        };
        assert_eq!(options[0].0, "zeta");
        assert_eq!(options[1].0, "alpha");
    }

    #[test]
    fn rejects_extra_fields_and_invalid_counts() {
        assert!(
            serde_json::from_value::<DecisionApiRequest>(json!({
                "model": "decision-latest",
                "state": "text",
                "questions": {},
                "stream": false
            }))
            .is_err()
        );
        let request: DecisionApiRequest = serde_json::from_value(json!({
            "model": "decision-latest",
            "state": "text",
            "questions": {
                "team": {"type": "choice", "instructions": "Pick", "criteria": {"only": null}}
            }
        }))
        .expect("schema parses before cardinality validation");
        assert!(request.into_prompt_request().is_err());
    }

    #[test]
    fn accepts_64_options_and_rejects_65() {
        let criteria = (0..64)
            .map(|index| (format!("option_{index}"), Value::Null))
            .collect::<serde_json::Map<_, _>>();
        let request: DecisionApiRequest = serde_json::from_value(json!({
            "model": "decision-latest",
            "state": "text",
            "questions": {
                "wide": {"type": "choice", "instructions": "Pick", "criteria": criteria}
            }
        }))
        .expect("64-option request schema");
        let prompt = request.into_prompt_request().expect("64-option request");
        let DecisionPromptQuestion::Choice { options, .. } = &prompt.questions[0].1 else {
            panic!("question is a choice");
        };
        assert_eq!(options.len(), 64);

        let criteria = (0..65)
            .map(|index| (format!("option_{index}"), Value::Null))
            .collect::<serde_json::Map<_, _>>();
        let request: DecisionApiRequest = serde_json::from_value(json!({
            "model": "decision-latest",
            "state": "text",
            "questions": {
                "wide": {"type": "choice", "instructions": "Pick", "criteria": criteria}
            }
        }))
        .expect("65-option request schema");
        assert!(request.into_prompt_request().is_err());
    }

    #[test]
    fn serializes_score_probabilities_and_legend() {
        let response = DecisionApiResponse::from_completion(
            "decision-1".to_string(),
            DecisionCompletion {
                answers: vec![(
                    "score".to_string(),
                    DecisionAnswer::Score {
                        score: 0.25,
                        legend: vec!["low".to_string(), "high".to_string()],
                        probabilities: vec![0.75, 0.25],
                        confidence: 0.2,
                    },
                )],
                usage: DecisionUsage {
                    input_tokens: 20,
                    output_tokens: 1,
                },
                timings: Default::default(),
                released_sequence_device_bytes: 0,
            },
            None,
        )
        .expect("valid completion");
        let value = serde_json::to_value(response).expect("serializable response");
        assert_eq!(value["answers"]["score"]["probabilities"]["1"], 0.25);
        assert_eq!(value["answers"]["score"]["legend"]["0"], "low");
    }

    #[test]
    fn validated_calibration_changes_probabilities_and_preserves_usage() {
        let artifact = json!({
            "schema": "eider-decision-calibration-v1",
            "validated_dataset": true,
            "model": "decision-1",
            "prompt_format": "eider-decision-v1",
            "target": "eider",
            "deployment": {
                "url": "http://127.0.0.1:8080/v1/decisions",
                "requested_model": "decision-latest",
                "returned_model": "decision-1"
            },
            "dataset": "pilot",
            "dataset_version": "1",
            "dataset_sha256": "0".repeat(64),
            "maps": {
                "noul": {"method": "temperature_nll", "temperature": 2.0},
                "choice": {"method": "temperature_nll", "temperature": 2.0},
                "score": {"method": "ordinal_temperature_rps", "temperature": 2.0}
            },
            "evaluation": {}
        });
        let bytes = serde_json::to_vec(&artifact).expect("artifact serializes");
        let calibration =
            DecisionCalibration::from_json(&bytes, "decision-1").expect("valid artifact");
        let response = DecisionApiResponse::from_completion(
            "decision-1".to_string(),
            DecisionCompletion {
                answers: vec![
                    (
                        "binary".to_string(),
                        DecisionAnswer::Noul { probability: 0.8 },
                    ),
                    (
                        "choice".to_string(),
                        DecisionAnswer::Choice {
                            choice: "a".to_string(),
                            probabilities: vec![("a".to_string(), 0.8), ("b".to_string(), 0.2)],
                            confidence: 0.5,
                        },
                    ),
                    (
                        "score".to_string(),
                        DecisionAnswer::Score {
                            score: 0.2,
                            legend: vec![
                                "low".to_string(),
                                "middle".to_string(),
                                "high".to_string(),
                            ],
                            probabilities: vec![0.8, 0.1, 0.1],
                            confidence: 0.5,
                        },
                    ),
                ],
                usage: DecisionUsage {
                    input_tokens: 9,
                    output_tokens: 2,
                },
                timings: Default::default(),
                released_sequence_device_bytes: 0,
            },
            Some(&calibration),
        )
        .expect("calibrated response");
        let value = serde_json::to_value(response).expect("response serializes");
        assert!((value["answers"]["binary"]["noul"].as_f64().unwrap() - 2.0 / 3.0).abs() < 1e-6);
        assert!(
            (value["answers"]["choice"]["probabilities"]["a"]
                .as_f64()
                .unwrap()
                - 2.0 / 3.0)
                .abs()
                < 1e-6
        );
        let score_probabilities = &value["answers"]["score"]["probabilities"];
        assert!(
            (score_probabilities["0"].as_f64().unwrap()
                - f64::from(8.0_f32.sqrt() / (8.0_f32.sqrt() + 2.0)))
            .abs()
                < 1e-6
        );
        assert!(value["answers"]["score"]["score"].as_f64().unwrap() > 0.4);
        assert_eq!(value["usage"]["input_tokens"], 9);
        assert_eq!(value["usage"]["output_tokens"], 2);
    }

    #[test]
    fn calibration_rejects_a_different_model() {
        let artifact = json!({
            "schema": "eider-decision-calibration-v1",
            "validated_dataset": true,
            "model": "decision-other",
            "prompt_format": "eider-decision-v1",
            "deployment": {
                "requested_model": "decision-latest",
                "returned_model": "decision-other"
            },
            "dataset": "pilot",
            "dataset_version": "1",
            "dataset_sha256": "0".repeat(64),
            "maps": {
                "noul": {"method": "temperature_nll", "temperature": 1.0},
                "choice": {"method": "temperature_nll", "temperature": 1.0},
                "score": {"method": "ordinal_temperature_rps", "temperature": 1.0}
            }
        });
        let bytes = serde_json::to_vec(&artifact).expect("artifact serializes");
        assert!(DecisionCalibration::from_json(&bytes, "decision-1").is_err());
    }
}
