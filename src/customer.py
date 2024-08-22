from client import Client
from order import Order
from manager import Manager
from warehouse import Warehouse

class Customer(Client):
    def __init__(self, client_id: int, name: str, email: str, phone_number: str, status: str, manager: Manager):
        super().__init__(client_id, name, email, phone_number)
        self.status = status
        self.manager = manager
        self.customer_orders = []
        self.warehouse_list = []
        manager.add_customer(self)

    def __repr__(self):
        return f"Customer:({self.client_id}, {self.name}, {self.status}, {self.manager})"


    def add_order(self, order):
        if order not in self.customer_orders:
            self.customer_orders.append(order)
            print (f"Order {order} added for customer {self.__repr__()}")
        else:
            print(f"Order {order} already exists for customer {self.__repr__()}")


    def show_customer_orders(self):
        print(f"Customer {self.__repr__()} has done orders below:\n"
              f"{self.customer_orders}")

    def add_warehouse(self, warehouse):
        if warehouse not in self.warehouse_list:
            self.warehouse_list.append(warehouse)
            print (f"Warehouse {warehouse} added to customer {self.__repr__()} warehouses list")
        else:
            print(f"Warehouse {warehouse} already is in customer{self.__repr__()} warehouses list")