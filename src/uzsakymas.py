# Assuming the following imports based on your given structure
from komercijosvadybininkas import KomercijosVadybininkas
from klientas import Klientas
from automobilis import Automobilis

class Uzsakymas:
    def __init__(self, uzsakymo_id: int, pakrovimo_data: str, atstumas: float, iskrovimo_data: str, pervezimo_kaina: float, automobilio_komplektacija: str, priekabos_tipas: str, klientas: Klientas, komercijos_vadybininkas: KomercijosVadybininkas, automobilis: Automobilis, uzsakymo_busena: str):
        self.uzsakymo_id = uzsakymo_id
        self.pakrovimo_data = pakrovimo_data
        self.atstumas = atstumas
        self.iskrovimo_data = iskrovimo_data
        self.pervezimo_kaina = pervezimo_kaina
        self.automobilio_komplektacija = automobilio_komplektacija
        self.priekabos_tipas = priekabos_tipas
        self.klientas = klientas
        self.komercijos_vadybininkas = komercijos_vadybininkas
        self.automobilis = automobilis
        self.uzsakymo_busena = uzsakymo_busena

    def sukurti_uzsakyma(self):
        # Logic for creating an order can be added here
        pass

    def gauti_informacija_apie_uzsakyma(self) -> str:
        print(f"Užsakymo ID: {self.uzsakymo_id}\n"
                f"Pakrovimo data: {self.pakrovimo_data}\n"
                f"Atstumas nuo pakrovimo iki iškrovimo: {self.atstumas}\n"
                f"Iškrovimo data: {self.iskrovimo_data}\n"
                f"Pervežimo kaina: {self.pervezimo_kaina}\n"
                f"Automobilio komplektacija: {self.automobilio_komplektacija}\n"
                f"Priekabos tipas: {self.priekabos_tipas}\n"
                f"Klientas: {self.klientas.pavadinimas}\n"
                f"Komercijos vadybininkas: {self.komercijos_vadybininkas}\n"
                f"Automobilis: {self.automobilis.automobilio_numeris}\n"
                f"Užsakymo būsena: {self.uzsakymo_busena}")

    def pasalinti_uzsakyma(self):
        # Logic for removing an order can be added here
        pass

    def redaguoti_uzsakymo_busena(self, nauja_busena: str):
        self.uzsakymo_busena = nauja_busena

    def keisti_komercijos_vadybininka(self, naujas_vadybininkas: KomercijosVadybininkas):
        self.komercijos_vadybininkas = naujas_vadybininkas
