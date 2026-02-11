# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel


# Canonical schemas (English keys)
# Prompts in any language should instruct the model to output JSON with exactly these keys.


class ExampleSituation(BaseModel):
    title: str
    scene: str


class ImplicitAssumption(BaseModel):
    assumption: str


class ExampleSituations(BaseModel):
    matching_situations: List[ExampleSituation]
    non_matching_situations: List[ExampleSituation]
    implicit_assumptions: List[ImplicitAssumption]
    segment: str


class ContextFreeReading(BaseModel):
    title: str
    description: str
    titles_of_example_situations_matching_this_reading: List[str]
    best_matching_example_situation: ExampleSituation
    commonalities_across_matching_example_situations: str
    differences_across_matching_example_situations: str


class ContextFreeReadings(BaseModel):
    readings: List[ContextFreeReading]


class SegmentVsContext(BaseModel):
    segment: str
    fit: Literal[
        "expected", 
        "surprising"
    ]
    rationale: str
    insight_gain: str


class SegmentVsPredictions(BaseModel):
    expected_segment: str
    actual_segment: str
    correspondence: Literal[
        "good", 
        "partial", 
        "poor_or_none"
    ]
    insight_gain: str


class SegmentVsPreviousCaseStructureHypothesis(BaseModel):
    confirmation: str
    challenge: str


class ContextFreeReadingVsContext(BaseModel):
    title: str
    fit: Literal[
        "good", 
        "partial", 
        "poor_or_none"
    ]
    rationale: str
    insight_gain: str


class Prediction(BaseModel):
    next_segment: str
    rationale: str


class ContextInformedReadingWithPrediction(BaseModel):
    title: str
    description: str
    plausibility: Literal[
        "very_plausible", 
        "plausible", 
        "only_just_plausible" # TODO: find a better expression here?
    ]
    most_plausible_prediction: Prediction
    only_just_plausible_prediction: Prediction


class ContextInformedReading(BaseModel):
    title: str
    description: str
    plausibility: Literal[
        "very_plausible", 
        "plausible", 
        "only_just_plausible" # TODO: find a better expression here?
    ]


class ConfrontationInputFirstRound(BaseModel):
    current_segment: str
    outer_context: str
    context_free_readings: ContextFreeReadings


class ConfrontationInput(BaseModel):
    current_segment: str
    outer_context: str
    inner_context: str
    predictions_for_next_segment_from_the_previous_round: List[str]
    hypothesis_from_the_previous_round: str
    context_free_readings: ContextFreeReadings


class ContextConfrontationFirstRound(BaseModel):
    segment_vs_context: SegmentVsContext
    context_free_readings_vs_context: List[ContextFreeReadingVsContext]
    interim_summary: str
    context_informed_readings_with_predictions_for_next_round: List[ContextInformedReadingWithPrediction]
    initial_case_structure_hypothesis: str


class ContextConfrontation(BaseModel):
    segment_vs_context: SegmentVsContext
    segment_vs_predictions_for_next_segment_from_the_previous_round: List[SegmentVsPredictions]
    segment_vs_previous_case_structure_hypothesis: SegmentVsPreviousCaseStructureHypothesis
    context_free_readings_vs_context: List[ContextFreeReadingVsContext]
    interim_summary: str
    context_informed_readings_with_predictions_for_next_round: List[ContextInformedReadingWithPrediction]
    updated_case_structure_hypothesis: str


class ContextConfrontationFinalRound(BaseModel):
    segment_vs_context: SegmentVsContext
    segment_vs_predictions_for_next_segment_from_the_previous_round: List[SegmentVsPredictions]
    segment_vs_previous_case_structure_hypothesis: SegmentVsPreviousCaseStructureHypothesis
    context_free_readings_vs_context: List[ContextFreeReadingVsContext]
    interim_summary: str
    context_informed_readings: List[ContextInformedReading]
    final_case_structure_hypothesis: str