# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel


### Schritt 1
#################################################################
class Beispielsituation(BaseModel):
    titel: str
    szene: str


# Keine Sonderzeichen hier. Schema geht direkt an OpenAI (Schritt 1)
class Beispielsituationen(BaseModel):
    beispielsituationen: List[Beispielsituation]


### Schritt 2

class KontextfreieLesart(BaseModel):
    titel: str
    beschreibung: str
    titel_der_zur_lesart_passenden_beispielsituationen: List[str]
    beste_zur_lesart_passende_beispielsituation: Beispielsituation
    gemeinsamekeiten_der_zur_lesart_passenden_beispielsituationen: str
    unterschiede_der_zur_lesart_passenden_beispielsituationen: str


# Keine Sonderzeichen hier. Schema geht direkt an OpenAI (Schritt 1)
class KontextfreieLesarten(BaseModel):
    lesarten: List[KontextfreieLesart]


### Schritt 3

class SequenzVsKontext(BaseModel):
    sequenz: str
    passung: Literal["erwartbar", "überraschend"]
    begründung: str
    erkenntnisgewinn: str


class SequenzVsErwarteteFortführung(BaseModel):
    erwartete_sequenz: str
    tatsächliche_sequenz: str
    entsprechung: Literal["gut", "teilweise", "schlecht/gar nicht"]
    erkenntnisgewinn: str


class SequenzVsAlteFallstrukturhypothese(BaseModel):
    bestätigung: str
    infragestellung: str


class KontextfreieLesartVsKontext(BaseModel):
    titel: str
    passung: Literal["gut", "teilweise", "schlecht/gar nicht"]
    begründung: str
    erkenntnisgewinn: str


class KontextinduzierteLesart(BaseModel):
    titel: str
    beschreibung: str


class KontextinduzierteLesarten(BaseModel):
    lesarten: List[KontextinduzierteLesart]


class PrognoseDerNächstenSequenzeinheit(BaseModel):
    lesart_titel: str
    nächste_sequenz: str
    begründung: str


class PrognoseDerNächstenSequenzeinheiten(BaseModel):
    prognose_pro_lesart: List[PrognoseDerNächstenSequenzeinheit]


# Variante A (erste Runde, mit angepasster Hypothese)
# Keine Sonderzeichen hier. Schema geht direkt an OpenAI (Schritt 3)
class KonfrontationMitKontextErsteRunde(BaseModel):
    sequenz_vs_kontext: SequenzVsKontext
    kontextfreie_lesarten_vs_kontext: List[KontextfreieLesartVsKontext]
    zwischenfazit: str
    kontextinduzierte_lesarten: KontextinduzierteLesarten
    prognose_der_nächsten_sequenzeinheit: PrognoseDerNächstenSequenzeinheit
    erste_fallstrukturhypothese: str


# Variante B (mittlere Runden)
# Keine Sonderzeichen hier. Schema geht direkt an OpenAI (Schritt 3)
class KonfrontationMitKontext(BaseModel):
    sequenz_vs_kontext: SequenzVsKontext
    sequenz_vs_erwartete_fortführung: List[SequenzVsErwarteteFortführung]
    sequenz_vs_alte_fallstrukturhypothese: SequenzVsAlteFallstrukturhypothese
    kontextfreie_lesarten_vs_kontext: List[KontextfreieLesartVsKontext]
    zwischenfazit: str
    kontextinduzierte_lesarten: KontextinduzierteLesarten
    prognose_der_nächsten_sequenzeinheit: PrognoseDerNächstenSequenzeinheit
    neue_fallstrukturhypothese: str


# Variante C (letzte Runde, ohne Fortführungen, dafür mit finaler Hypothese)
# Keine Sonderzeichen hier. Schema geht direkt an OpenAI (Schritt 3)
class KonfrontationMitKontextLetzteRunde(BaseModel):
    sequenz_vs_kontext: SequenzVsKontext
    sequenz_vs_erwartete_fortführung: List[SequenzVsErwarteteFortführung]
    sequenz_vs_alte_fallstrukturhypothese: SequenzVsAlteFallstrukturhypothese
    kontextfreie_lesarten_vs_kontext: List[KontextfreieLesartVsKontext]
    zwischenfazit: str
    kontextinduzierte_lesarten: KontextinduzierteLesarten
    finale_fallstrukturhypothese: str