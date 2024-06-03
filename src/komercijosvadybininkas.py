from darbuotuojas import Darbuotojas

class KomercijosVadybininkas(Darbuotuojas):
    def __init__(self, vardas, pavarde, asmens_kodas, telefono_numeris, elektroninis_pastas, alga):
        super().__init__(vardas, pavarde, asmens_kodas, telefono_numeris, elektroninis_pastas, alga)
        self.aptarnaujami_klientai = []
        self.uzsakymai = []

    def perziureti_vadybininko_priziurimus_klientus(self):
        return self.aptarnaujami_klientai

    def priskirti_nauja_klienta(self, klientas):
        self.aptarnaujami_klientai.append(klientas)

    def atimti_nauja_klienta(self, klientas):
        self.aptarnaujami_klientai.remove(klientas)

    def perziureti_vadybininko_priziurimus_uzsakymus(self):
        return self.uzsakymai