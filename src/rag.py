# Import libraries
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_title: str
    section: Optional[str] = None
    chunk_id: str


class AnswerSection(BaseModel):
    text: str
    citations: List[Citation] = Field(default_factory=list)


class MedAnswer(BaseModel):
    question: str
    short_answer: AnswerSection
    major_warnings: List[AnswerSection] = Field(default_factory=list)
    interactions: List[AnswerSection] = Field(default_factory=list)
    monitoring: List[AnswerSection] = Field(default_factory=list)
    counseling_points: List[AnswerSection] = Field(default_factory=list)
    limitations: Optional[str] = None


@dataclass
class Chunk:
    chunk_id: str
    drug_name: str
    source_title: str
    section: str
    text: str


class SimpleRAGEngine:
    """
    Starter RAG engine.

    This is intentionally lightweight for a portfolio MVP:
    - uses a tiny in-memory corpus
    - uses keyword matching as a stand-in for real retrieval
    - returns structured output for the UI

    Replace `retrieve()` with embeddings + FAISS when you ingest real docs.
    """

    def __init__(self) -> None:
        self.corpus = [
            Chunk(
                chunk_id="warfarin_fda_001",
                drug_name="warfarin",
                source_title="Warfarin FDA Label",
                section="Drug Interactions",
                text=(
                    "Warfarin has clinically significant interactions with many drugs and foods. "
                    "Concomitant use may increase or decrease INR, requiring close monitoring."
                ),
            ),
            Chunk(
                chunk_id="apixaban_fda_001",
                drug_name="apixaban",
                source_title="Apixaban FDA Label",
                section="Warnings and Precautions",
                text=(
                    "Apixaban carries bleeding risk. Clinical monitoring should include signs of bleeding "
                    "and considerations around renal function and peri-procedural interruption."
                ),
            ),
            Chunk(
                chunk_id="rivaroxaban_fda_001",
                drug_name="rivaroxaban",
                source_title="Rivaroxaban FDA Label",
                section="Patient Counseling Information",
                text=(
                    "Patients should be counseled on bleeding risk, adherence, and when to seek urgent care."
                ),
            ),
            Chunk(
                chunk_id="dabigatran_fda_001",
                drug_name="dabigatran",
                source_title="Dabigatran FDA Label",
                section="Patient Counseling Information",
                text=(
                    "Counsel patients not to break, chew, or empty capsules and to store the medication "
                    "in the original container when applicable."
                ),
            ),
            Chunk(
                chunk_id="enoxaparin_fda_001",
                drug_name="enoxaparin",
                source_title="Enoxaparin FDA Label",
                section="Boxed Warning",
                text=(
                    "Epidural or spinal hematomas may occur in patients receiving enoxaparin with neuraxial procedures."
                ),
            ),
        ]

    def classify_question(self, question: str) -> str:
        q = question.lower()
        if "compare" in q or " vs " in q:
            return "comparison"
        if "interaction" in q:
            return "interaction"
        if "monitor" in q:
            return "monitoring"
        if "counsel" in q:
            return "counseling"
        return "summary"

    def retrieve(self, question: str, top_k: int = 3) -> List[Chunk]:
        q = question.lower()
        scored = []
        for chunk in self.corpus:
            score = 0
            for term in [chunk.drug_name.lower(), chunk.section.lower()]:
                if term in q:
                    score += 2
            for token in q.split():
                if token in chunk.text.lower():
                    score += 1
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [chunk for score, chunk in scored if score > 0][:top_k]
        return results or self.corpus[:top_k]

    def answer(self, question: str) -> MedAnswer:
        retrieved = self.retrieve(question)
        citations = [
            Citation(
                source_title=chunk.source_title,
                section=chunk.section,
                chunk_id=chunk.chunk_id,
            )
            for chunk in retrieved
        ]

        short_answer_text = " ".join(chunk.text for chunk in retrieved[:2])

        return MedAnswer(
            question=question,
            short_answer=AnswerSection(text=short_answer_text, citations=citations[:2]),
            major_warnings=[
                AnswerSection(
                    text="Bleeding risk and procedure-related precautions are recurring themes in the indexed starter documents.",
                    citations=citations[:1],
                )
            ],
            interactions=[
                AnswerSection(
                    text="Warfarin has multiple clinically significant interactions and requires close INR-related monitoring when therapy changes.",
                    citations=[c for c in citations if "warfarin" in c.chunk_id][:1],
                )
            ],
            monitoring=[
                AnswerSection(
                    text="Monitoring considerations include bleeding signs, medication changes, and context-specific renal or procedural considerations.",
                    citations=citations[:2],
                )
            ],
            counseling_points=[
                AnswerSection(
                    text="Counsel on adherence, bleeding symptoms, and product-specific administration instructions when present in the source material.",
                    citations=citations[-2:],
                )
            ],
            limitations=(
                "This starter app uses a tiny demo corpus and simple retrieval. Replace with real source ingestion, "
                "embeddings, and evaluation before presenting results as robust evidence summaries."
            ),
        )
