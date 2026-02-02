# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel


### German classes
###############################################################################

class Beispielsituation(BaseModel):
    titel: str
    szene: str

class ImpliziteVoraussetzungen(BaseModel):
    annahme: str

class Beispielsituationen(BaseModel):
    passende_situationen: List[Beispielsituation]
    unpassende_situationen: List[Beispielsituation]
    implizite_voraussetzungen: List[ImpliziteVoraussetzungen]

class KontextfreieLesart(BaseModel):
    titel: str
    beschreibung: str
    titel_der_zur_lesart_passenden_beispielsituationen: List[str]
    beste_zur_lesart_passende_beispielsituation: Beispielsituation
    gemeinsamekeiten_der_zur_lesart_passenden_beispielsituationen: str
    unterschiede_der_zur_lesart_passenden_beispielsituationen: str

class KontextfreieLesarten(BaseModel):
    lesarten: List[KontextfreieLesart]

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

class Prognose(BaseModel):
    nächste_sequenz: str
    begründung: str

class KontextinduzierteLesartMitPrognose(BaseModel):
    titel: str
    beschreibung: str
    prognose_am_wahrscheinlichsten: Prognose
    prognose_gerade_noch_im_Rahmen: Prognose

class KontextinduzierteLesart(BaseModel):
    titel: str
    beschreibung: str    

class KonfrontationMitKontextErsteRunde(BaseModel):
    sequenz_vs_kontext: SequenzVsKontext
    kontextfreie_lesarten_vs_kontext: List[KontextfreieLesartVsKontext]
    zwischenfazit: str
    kontextinduzierte_lesarten_mit_prognose_für_die_nächste_runde: List[KontextinduzierteLesartMitPrognose]
    erste_fallstrukturhypothese: str

class KonfrontationMitKontext(BaseModel):
    sequenz_vs_kontext: SequenzVsKontext
    sequenz_vs_erwartete_fortführung: List[SequenzVsErwarteteFortführung]
    sequenz_vs_alte_fallstrukturhypothese: SequenzVsAlteFallstrukturhypothese
    kontextfreie_lesarten_vs_kontext: List[KontextfreieLesartVsKontext]
    zwischenfazit: str
    kontextinduzierte_lesarten_mit_prognose_für_die_nächste_runde: List[KontextinduzierteLesartMitPrognose]
    neue_fallstrukturhypothese: str

class KonfrontationMitKontextLetzteRunde(BaseModel):
    sequenz_vs_kontext: SequenzVsKontext
    sequenz_vs_erwartete_fortführung: List[SequenzVsErwarteteFortführung]
    sequenz_vs_alte_fallstrukturhypothese: SequenzVsAlteFallstrukturhypothese
    kontextfreie_lesarten_vs_kontext: List[KontextfreieLesartVsKontext]
    zwischenfazit: str
    kontextinduzierte_lesarten: List[KontextinduzierteLesart]
    finale_fallstrukturhypothese: str
