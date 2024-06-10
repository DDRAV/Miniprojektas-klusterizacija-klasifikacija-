from darbuotuojas import Darbuotojas

class TransportoVadybininkas(Darbuotojas):
    def __init__(self, vardas, pavarde, asmens_kodas, telefono_numeris, elektroninis_pastas, alga, id):
        super().__init__(vardas, pavarde, asmens_kodas, telefono_numeris, elektroninis_pastas, alga, id)
        self.priziurimas_transportas = []

    def __repr__(self):
        return f"Komercijos Vadybyninkas({self.vardas}, {self.pavarde}, {self.elektroninis_pastas}, {self.id})"

    def perziureti_transporta(self):
        print(f"Vadybininkas {self.vardas} {self.pavarde} priziuri {self.priziurimas_transportas} automobilius")

    def priskirti_automobili(self, automobilis):
        if automobilis not in self.priziurimas_transportas:
            self.priziurimas_transportas.append(automobilis)
            print(f"Automobilis {automobilis.automobilio_numeris} priskirtas vadybininkui {self.vardas} {self.pavarde}.")

    def atimti_automobili(self, automobilis):
        if automobilis in self.priziurimas_transportas:
            self.priziurimas_transportas.remove(automobilis)
            print(f"Automobilis {automobilis.automobilio_numeris} atimtas iš vadybininko {self.vardas} {self.pavarde}.")
