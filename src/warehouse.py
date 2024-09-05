from customer import Customer


class Warehouse:
    def __init__(self, wh_id: int, name: str, customer: Customer, country: str, postal_code: str, adress: str,
                 working_hours: int, type: str):
        self.wh_id = wh_id
        self.name = name
        self.customer = customer
        self.country = country
        self.postal_code = postal_code
        self.adress = adress
        self.working_hours = working_hours
        self.type = type
        customer.add_warehouse(self)

    def __repr__(self):
        return f"Warehouse({self.wh_id}, {self.name}, {self.type}, {self.customer})"

    def wh_info(self):
        print(f"Warehouse ID: {self.wh_id}\n"
              f"Name: {self.name}\n"
              f"Customer: {self.customer}\n"
              f"Country: {self.country}\n"
              f"Postal code: {self.postal_code}\n"
              f"Adress: {self.adress}\n"
              f"Working hours: {self.working_hours}\n"
              f"Type: {self.type}\n")
