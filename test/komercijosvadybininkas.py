from darbuotuojas import Darbuotojas

class KomercijosVadybininkas(Darbuotojas):
    def __init__(self, vardas, pavarde, asmens_kodas, telefono_numeris, elektroninis_pastas, alga, id):
        super().__init__(vardas, pavarde, asmens_kodas, telefono_numeris, elektroninis_pastas, alga, id)
        self.aptarnaujami_klientai = []
        self.uzsakymai = []

    def __repr__(self):
        return f"Komercijos Vadybyninkas({self.vardas}, {self.pavarde}, {self.elektroninis_pastas}, {self.id})"

    def perziureti_klientus(self):
        print(f"Vadybininkas {self.vardas} {self.pavarde} aptarnauja {self.aptarnaujami_klientai} klientus")

    def priskirti_klienta(self, klientas):
        self.aptarnaujami_klientai.append(klientas)

    def atimti_klienta(self, klientas):
        self.aptarnaujami_klientai.remove(klientas)

    def perziureti_uzsakymus(self):
        print(f"Vadybininkas {self.vardas} {self.pavarde} priziuri {self.uzsakymai} uzsakymus")

    def prideti_uzsakyma_kv(self, uzsakymas):
        self.uzsakymai.append(uzsakymas)
        print(f"Užsakymas pridėtas komercijos vadybininkui {self.vardas}: {uzsakymas.uzsakymo_id}")