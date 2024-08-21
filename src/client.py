class Client:
    def __init__(self, client_id: int, name: str, email: str, phone_number: str):
        self.client_id = client_id
        self.name = name
        self.email = email
        self.phone_number = phone_number


    def __repr__(self):
        return f"Client({self.client_id}, {self.name})"

    def client_info(self):
        print(f"Client ID: {self.client_id}\n"
                f"Name: {self.name}\n"
                f"Email: {self.email}\n"
                f"Phone number: {self.phone_number}\n")

#    def uzsakymu_info(self):
 #       if not self.kliento_uzsakymai:
  #          return "Klientas neturi užsakymų"
#
 #       uzsakymai_info = "Kliento užsakymų sąrašas:\n"
  #      for indeksas, uzsakymas in enumerate(self.kliento_uzsakymai, start=1):
   #         uzsakymai_info += f"Užsakymas {indeksas}: ID {uzsakymas.uzsakymo_id}\n"
    #    print(uzsakymai_info)
#
 #   def prideti_uzsakyma(self, uzsakymas):
  #      self.kliento_uzsakymai.append(uzsakymas)