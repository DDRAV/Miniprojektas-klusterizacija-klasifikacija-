from komercijosvadybininkas import KomercijosVadybininkas

class Klientas:
    def __init__(self, pavadinimas: str, kontaktinis_numeris: str, elektroninis_pastas: str, klientas_id: int,
                 komercijos_vadybininkas: KomercijosVadybininkas):
        self.pavadinimas = pavadinimas
        self.kontaktinis_numeris = kontaktinis_numeris
        self.elektroninis_pastas = elektroninis_pastas
        self.klientas_id = klientas_id
        self.kliento_uzsakymai = []
        self.komercijos_vadybininkas = komercijos_vadybininkas

    def get_all_information(self):
        print(f"Klientas: {self.pavadinimas}\n"
                f"Kontaktinis numeris: {self.kontaktinis_numeris}\n"
                f"Elektroninis paštas: {self.elektroninis_pastas}\n"
                f"Kliento ID: {self.klientas_id}\n"
                f"Komercijos vadybininkas: {self.komercijos_vadybininkas}\n"
                f"Užsakymų kiekis: {len(self.kliento_uzsakymai)}")

    def get_order_information(self):
        if not self.kliento_uzsakymai:
            return "Klientas neturi užsakymų"

        uzsakymai_info = "Kliento užsakymų sąrašas:\n"
        for indeksas, uzsakymas in enumerate(self.kliento_uzsakymai, start=1):
            uzsakymai_info += f"Užsakymas {indeksas}: ID {uzsakymas.uzsakymo_id}\n"
        print(uzsakymai_info)

    def add_uzsakymas(self, uzsakymas):
        self.kliento_uzsakymai.append(uzsakymas)