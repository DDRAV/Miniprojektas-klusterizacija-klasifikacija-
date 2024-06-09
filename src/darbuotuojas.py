class Darbuotojas:
    def __init__(self, vardas: str, pavarde: str, asmens_kodas: int, telefono_numeris: str, elektroninis_pastas: str, alga: int, id: str):
        self.vardas = vardas
        self.pavarde = pavarde
        self.asmens_kodas = asmens_kodas
        self.telefono_numeris = telefono_numeris
        self.elektroninis_pastas = elektroninis_pastas
        self.alga = alga
        self.id = id

    def gauti_informacija(self):
        print(f"Vardas: {self.vardas}\n"
            f"Pavarde: {self.pavarde}\n"
            f"Asmens_kodas: {self.asmens_kodas}\n"
            f"Telefono_numeris: {self.telefono_numeris}\n"
            f"Elektroninis_pastas: {self.elektroninis_pastas}\n"
            f"Alga: {self.alga}\n"
            f"ID: {self.id}\n")

    def keisti_informacija(self, vardas=None, pavarde=None, telefono_numeris=None, elektroninis_pastas=None, alga=None):
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