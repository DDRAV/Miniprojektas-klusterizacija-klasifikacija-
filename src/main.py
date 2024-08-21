from automobilis import Automobilis
from client import Client
from komercijosvadybininkas import KomercijosVadybininkas
from transportovadybininkas import TransportoVadybininkas
from order import Uzsakymas

komercijos_vadybininkas = KomercijosVadybininkas(
    vardas="Jonas", pavarde="Jonaitis", asmens_kodas=12345678901,
    telefono_numeris="+37060012345", elektroninis_pastas="jonas@example.com", alga=1500, id= "KV1"
)

transporto_vadybininkas = TransportoVadybininkas(
    vardas="Petras", pavarde="Petraitis", asmens_kodas=10987654321,
    telefono_numeris="+37060054321", elektroninis_pastas="petras@example.com", alga=1700, id= "TV1"
)
transporto_vadybininkas2 = TransportoVadybininkas(
    vardas="Antanas", pavarde="Antanaitis", asmens_kodas=12121212121,
    telefono_numeris="+37067777777", elektroninis_pastas="antanas@example.com", alga=1600, id= "TV2"
)
klientas = Klientas(
    pavadinimas="ABC Įmonė", kontaktinis_numeris="+37061234567",
    elektroninis_pastas="info@abcimone.lt", klientas_id=1001,
    komercijos_vadybininkas=komercijos_vadybininkas
)

uzsakymas1 = Uzsakymas(uzsakymo_id=1,pakrovimo_data="2024.05.20",
                       atstumas=1234, iskrovimo_data="2024.05.23",
                       pervezimo_kaina= 1400,automobilio_komplektacija="Standartine",
                       priekabos_tipas="Tentas",klientas=klientas,
                       komercijos_vadybininkas=komercijos_vadybininkas,uzsakymo_busena="Sukurta"
                       )

# Priskiriam uzsakyma prie kliento

klientas.prideti_uzsakyma(uzsakymas1)

# Komercijos vadybininkas priskiria klientą
komercijos_vadybininkas.priskirti_klienta(klientas)

# Sukuriame automobilį ir priskiriame užsakymą ir pakeiciame uzsakymo busena
automobilis = Automobilis(
    automobilio_numeris="ABC123", automobilio_komplektacija="Standartinė",
    priekabos_numeris="PRK456", priekabos_tipas="Šaldytuvas",
    vairuotojo_kontaktinis_tel="+37065555555", vygdomas_uzsakymas=uzsakymas1,
    automobilio_busena="Parengtas", atsakingas_tv=transporto_vadybininkas
)

uzsakymas1.redaguoti_uzsakymo_busena("Suplanuota")
# Informacijos atvaizdavimas
print("Informacija apie komercijos vadybininką ir jo klientus:")
komercijos_vadybininkas.perziureti_klientus()

print("\nInformacija apie transporto vadybininką ir jo automobilius:")
automobilis.gauti_informacija_apie_automobili()


Uzsakymas.prideti_uzsakyma(
    uzsakymo_id=1,
    pakrovimo_data="2024-06-10",
    atstumas=100.0,
    iskrovimo_data="2024-06-11",
    pervezimo_kaina=500.0,
    automobilio_komplektacija="Standartine",
    priekabos_tipas="Saldytuvas",
    klientas=klientas,
    komercijos_vadybininkas=komercijos_vadybininkas,
    uzsakymo_busena="Pending"
)

print("\nInformacija apie klientą ir jo užsakymus:")
klientas.bendra_info()
klientas.uzsakymu_info()

# Pakeitimai
print("\nAtliekami pakeitimai:")
automobilis.keisti_automobilio_komplektacija("ADR")
automobilis.keisti_priekabos_numeri("PRK789")
automobilis.keisti_priekabos_tipa("Tentinis")
automobilis.keisti_atsakinga_tv(transporto_vadybininkas2)

print("\nAtnaujinta informacija apie automobilį:")
automobilis.gauti_informacija_apie_automobili()


if __name__ == "__main__":
    conn = db_engine.connect_db()
    if conn:
        # Perform your database operations here
        pass
    db_engine.close_db(conn)