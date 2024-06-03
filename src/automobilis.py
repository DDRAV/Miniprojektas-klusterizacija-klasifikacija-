from transportovadybininkas import TransportoVadybininkas
from uzsakymas import Uzsakymas

class Automobilis:
    def __init__(self, automobilio_numeris: str, automobilio_komplektacija: str, priekabos_numeris: str, priekabos_tipas: str, vairuotojo_kontaktinis_tel: str, vygdomas_uzsakymas: Uzsakymas, automobilio_busena: str, atsakingas_tv: TransportoVadybininkas):
        self.automobilio_numeris = automobilio_numeris
        self.automobilio_komplektacija = automobilio_komplektacija
        self.priekabos_numeris = priekabos_numeris
        self.priekabos_tipas = priekabos_tipas
        self.vairuotojo_kontaktinis_tel = vairuotojo_kontaktinis_tel
        self.vygdomas_uzsakymas = vygdomas_uzsakymas
        self.automobilio_busena = automobilio_busena
        self.atsakingas_tv = atsakingas_tv

    def gauti_informacija_apie_automobili(self) -> str:
        return (f"Automobilio numeris: {self.automobilio_numeris}\n"
                f"Automobilio komplektacija: {self.automobilio_komplektacija}\n"
                f"Priekabos numeris: {self.priekabos_numeris}\n"
                f"Priekabos tipas: {self.priekabos_tipas}\n"
                f"Vairuotojo kontaktinis tel: {self.vairuotojo_kontaktinis_tel}\n"
                f"Vygdomas užsakymas: {self.vygdomas_uzsakymas}\n"
                f"Automobilio būsenа: {self.automobilio_busena}\n"
                f"Atsakingas TV: {self.atsakingas_tv}")

    def keisti_atsakinga_tv(self, naujas_tv: TransportoVadybininkas):
        self.atsakingas_tv = naujas_tv

    def keisti_automobilio_komplektacija(self, nauja_komplektacija: str):
        self.automobilio_komplektacija = nauja_komplektacija

    def keisti_priekabos_numeri(self, naujas_priekabos_numeris: str):
        self.priekabos_numeris = naujas_priekabos_numeris

    def keisti_priekabos_tipa(self, naujas_priekabos_tipas: str):
        self.priekabos_tipas = naujas_priekabos_tipas

    def priskirti_nauja_uzsakyma(self, naujas_uzsakymas: Uzsakymas):
        self.vygdomas_uzsakymas = naujas_uzsakymas