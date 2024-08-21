from warehouse import Warehouse

class Order:
    orders = []
    def __init__(self, uzsakymo_id: int, pakrovimo_data: str, atstumas: float, iskrovimo_data: str, pervezimo_kaina: float, automobilio_komplektacija: str, priekabos_tipas: str, klientas: Klientas, komercijos_vadybininkas: KomercijosVadybininkas, uzsakymo_busena: str):
        self.uzsakymo_id = uzsakymo_id
        self.pakrovimo_data = pakrovimo_data
        self.atstumas = atstumas
        self.iskrovimo_data = iskrovimo_data
        self.pervezimo_kaina = pervezimo_kaina
        self.automobilio_komplektacija = automobilio_komplektacija
        self.priekabos_tipas = priekabos_tipas
        self.klientas = klientas
        self.komercijos_vadybininkas = komercijos_vadybininkas
        self.uzsakymo_busena = uzsakymo_busena
        klientas.prideti_uzsakyma(self)
        komercijos_vadybininkas.prideti_uzsakyma_kv(self)


    @classmethod
    def prideti_uzsakyma(cls, uzsakymo_id, pakrovimo_data, atstumas, iskrovimo_data, pervezimo_kaina, automobilio_komplektacija,
                      priekabos_tipas, klientas, komercijos_vadybininkas, uzsakymo_busena):
        new_order = cls(uzsakymo_id, pakrovimo_data, atstumas, iskrovimo_data, pervezimo_kaina, automobilio_komplektacija,
                        priekabos_tipas, klientas, komercijos_vadybininkas, uzsakymo_busena)
        cls.uzsakymai.append(new_order)
        klientas.prideti_uzsakyma(new_order)
        komercijos_vadybininkas.prideti_uzsakyma_kv(new_order)
        print(f"Naujas užsakymas pridėtas: {new_order.uzsakymo_id}")

    def __repr__(self):
        return f"Uzsakymas({self.uzsakymo_id}, {self.klientas}, {self.komercijos_vadybininkas}, {self.uzsakymo_busena})"

    def gauti_informacija_apie_uzsakyma(self):
        print(f"Užsakymo ID: {self.uzsakymo_id}\n"
                f"Pakrovimo data: {self.pakrovimo_data}\n"
                f"Atstumas nuo pakrovimo iki iškrovimo: {self.atstumas}\n"
                f"Iškrovimo data: {self.iskrovimo_data}\n"
                f"Pervežimo kaina: {self.pervezimo_kaina}\n"
                f"Automobilio komplektacija: {self.automobilio_komplektacija}\n"
                f"Priekabos tipas: {self.priekabos_tipas}\n"
                f"Klientas: {self.klientas.pavadinimas}\n"
                f"Komercijos vadybininkas: {self.komercijos_vadybininkas}\n"
                f"Užsakymo būsena: {self.uzsakymo_busena}")

    def redaguoti_uzsakymo_busena(self, nauja_busena: str):
        self.uzsakymo_busena = nauja_busena
        print(f"Užsakymo {self.uzsakymo_id} būsena pakeista į {nauja_busena}.")

    def keisti_komercijos_vadybininka(self, naujas_vadybininkas):
        self.komercijos_vadybininkas.uzsakymai.remove(self)
        self.komercijos_vadybininkas = naujas_vadybininkas
        naujas_vadybininkas.prideti_uzsakyma(self)
        print(f"Užsakymo {self.uzsakymo_id} komercijos vadybininkas pakeistas į {naujas_vadybininkas.vardas}.")