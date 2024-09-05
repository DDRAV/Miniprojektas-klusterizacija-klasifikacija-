from client import Client


class Carrier(Client):
    def __init__(self, client_id: int, name: str, email: str, phone_number: str, country_of_reg: str):
        super().__init__(client_id, name, email, phone_number)
        self.country_of_reg = country_of_reg
        self.assembled_orders = []


    def __repr__(self):
        return f"Carrier({self.client_id}, {self.name}, {self.email}, {self.phone_number}, {self.country_of_reg})"


    def add_order(self, order):
        if order not in self.assembled_orders:
            self.assembled_orders.append(order)
            print (f"Order {order} added for carrier {self.__repr__()}")
        else:
            print(f"Order {order} already exists for carrier {self.__repr__()}")

    def show_assembled_orders(self):
        print(f"Carrier {self.__repr__()} has carried orders below:\n"
              f"{self.assembled_orders}")


    def remove_order(self, order):
        if order in self.assembled_orders:
            self.assembled_orders.remove(order)
            print (f"Order {order} removed from carrier {self.__repr__()}")
        else:
            print(f"Order {order} wasn't carried by carrier {self.__repr__()}")
