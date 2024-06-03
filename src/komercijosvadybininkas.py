from darbuotuojas import Darbuotojas

class KomercijosVadybininkas(Darbuotojas):
    def __init__(self, vardas, pavarde, asmens_kodas, telefono_numeris, elektroninis_pastas, alga):
        super().__init__(vardas, pavarde, asmens_kodas, telefono_numeris, elektroninis_pastas, alga)
        self.aptarnaujami_klientai = []
        self.uzsakymai = []

    def perziureti_klientus(self):
        print(f"Vadybininkas {self.vardas} {self.pavarde} aptarnauja {self.aptarnaujami_klientai} klientus")

    def priskirti_klienta(self, klientas):
        self.aptarnaujami_klientai.append(klientas)

    def atimti_klienta(self, klientas):
        self.aptarnaujami_klientai.remove(klientas)

    def perziureti_uzsakymus(self):
        print(f"Vadybininkas {self.vardas} {self.pavarde} priziuri {self.uzsakymai} uzsakymus")