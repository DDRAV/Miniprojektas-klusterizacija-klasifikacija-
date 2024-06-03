class Darbuotojas:
    def __init__(self, vardas: str, pavarde: str, asmens_kodas: int, telefono_numeris: str, elektroninis_pastas: str, alga: str):
        self.vardas = vardas
        self.pavarde = pavarde
        self.asmens_kodas = asmens_kodas
        self.telefono_numeris = telefono_numeris
        self.elektroninis_pastas = elektroninis_pastas
        self.alga = alga

    def gauti_bendra_informacija(self):
        return {
            "Vardas": self.vardas,
            "Pavarde": self.pavarde,
            "Asmens_kodas": self.asmens_kodas,
            "Telefono_numeris": self.telefono_numeris,
            "Elektroninis_pastas": self.elektroninis_pastas,
            "Alga": self.alga
        }

    def keisti_bendra_informacija(self, vardas=None, pavarde=None, telefono_numeris=None, elektroninis_pastas=None, alga=None):
        if vardas:
            self.vardas = vardas
        if pavarde:
            self.pavarde = pavarde
        if telefono_numeris:
            self.telefono_numeris = telefono_numeris
        if elektroninis_pastas:
            self.elektroninis_pastas = elektroninis_pastas
        if alga:
            self.alga = alga