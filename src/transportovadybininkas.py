from darbuotuojas import Darbuotojas

class TransportoVadybininkas(Darbuotojas):
    def __init__(self, vardas, pavarde, asmens_kodas, telefono_numeris, elektroninis_pastas, alga, id):
        super().__init__(vardas, pavarde, asmens_kodas, telefono_numeris, elektroninis_pastas, alga, id)
        self.priziurimas_transportas = []

    def perziureti_transporta(self):
        print(f"Vadybininkas {self.vardas} {self.pavarde} priziuri {self.priziurimas_transportas} automobilius")

    def priskirti_automobili(self, automobilis):
        self.priziurimas_transportas.append(automobilis)

    def atimti_automobili(self, automobilis):
        self.priziurimas_transportas.remove(automobilis)
