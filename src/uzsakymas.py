from datetime import date
from darbuotuojas import Darbuotojas
from klientas import Klientas
from komercijosvadybininkas import KomercijosVadybininkas

class Uzsakymas:
    def __init__(self, uzsakymo_id: str, pakrovimo_data: date, iskroviomo_data: date, atstumas: float, pervezimo_kaina: float, automobilio_komplektacija: str, priekabos_tipas: str, klientas_id: int, komercios_vadybininkas_id: int, busena: str = "Sukurta", automobilis_id=None):
        self.uzsakymo_id = uzsakymo_id
        self.pakrovimo_data = pakrovimo_data
        self.iskroviomo_data = iskroviomo_data
        self.atstumas = atstumas
        self.pervezimo_kaina = pervezimo_kaina
        self.automobilio_komplektacija = automobilio_komplektacija
        self.priekabos_tipas = priekabos_tipas
        self.klientas_id = klientas_id
        self.komercijos_vadybininkas_id = komercios_vadybininkas_id
        self.busena = busena
        self.automobilis_id = automobilis_id
        self.klientas = None
        self.komercijos_vadybininkas = None

        self.set_klientas(klientas_id)
        self.set_komercijos_vadybininkas(id)

    def __str__(self):
        return f"""
        Užsakymo ID: {self.uzsakymo_id}
        Pakrovimo data: {self.pakrovimo_data}
        Iškrovimo data: {self.iskroviomo_data}
        Atstumas: {self.atstumas:.2f} km
        Pervežimo kaina: {self.pervežimo_kaina:.2f}€
        Automobilio komplektacija: {self.automobilio_komplektacija}
        Priekabos tipas: {self.priekabos_tipas}
        Klientas: {self.klientas.kliento_pavadinimas} ({self.klientas.kliento_id})
        Komercijos vadybininkas: {self.komercijos_vadybininkas.vardas} {self.komercijos_vadybininkas.pavarde} ({self.komercijos_vadybininkas.id})
        Būsena: {self.busena}
        """

    def sukurti_uzsakyma(self):
        # Implement logic to create the order (e.g., save to a database)
        print(f"Užsakymas {self.uzsakymo_id} sukurtas")

    def gauti_informacija_apie_uzsakyma(self):
        return {
            "Uzsakymo_id": self.uzsakymo_id,
            "Pakrovimo_data": self.pakrovimo_data.strftime("%Y-%m-%d"),
            "Iskroviomo_data": self.iskroviomo_data.strftime("%Y-%m-%d"),
            "Atstumas": self.atstumas,
            "Pervežimo_kaina": self.pervezimo_kaina,
            "Automobilio_komplektacija": self.automobilio_komplektacija,
            "Priekabos_tipas": self.priekabos_tipas,
            "Klientas": self.klientas.kliento_info(),
            "Komercijos_vadybininkas": self.komercijos_vadybininkas