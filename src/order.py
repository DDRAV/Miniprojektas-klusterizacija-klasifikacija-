from warehouse import Warehouse
from customer import Customer
from carrier import Carrier
from manager import Manager

class Order:
    orders = []
    def __init__(self, order_id: int, price: int, loading_wh: Warehouse, loading_time: int, unloading_wh: Warehouse, unloading_time: int, manager: Manager, customer: Customer, assembled_carrier: Carrier):
        self.order_id = order_id
        self.price = price
        self.loading_wh = loading_wh
        self.loading_time = loading_time
        self.unloading_wh = unloading_wh
        self.unloading_time = unloading_time
        self.manager = manager
        manager.add_order(self)
        self.customer = customer
        customer.add_order(self)
        self.assembled_carrier = None


    @classmethod
    def prideti_uzsakyma(cls, order_id, price, loading_wh, loading_time, unloading_wh, unloading_time,
                      manager, customer, assembled_carrier):
        new_order = cls(order_id, price, loading_wh, loading_time, unloading_wh, unloading_time,
                      manager, customer,assembled_carrier)
        cls.orders.append(new_order)
        manager.add_order(new_order)
        customer.add_order(new_order)
        print(f"New order {new_order.order_id} is made")

    def __repr__(self):
        return f"Order({self.order_id}, {self.price}, {self.manager}, {self.customer}, {self.assembled_carrier})"

    def order_info(self):
        print(f"Order ID: {self.order_id}\n"
                f"Price: {self.price}\n"
                f"Loading warehouse: {self.loading_wh}\n"
                f"Loading time: {self.loading_time}\n"
                f"Unloading warehouse: {self.unloading_wh}\n"
                f"Unloading time: {self.unloading_time}\n"
                f"Manager: {self.manager}\n"
                f"Customer: {self.customer}\n"
                f"Assembled carrier: {self.assembled_carrier}\n")

    def change_carrier(self, new_carrier):
        if self.assembled_carrier is None:
            self.assembled_carrier = new_carrier
            print(f"Order {self.order_id} carrier {new_carrier} has been assigned.")
        else:
            self.assembled_carrier.remove_order(self)
            self.assembled_carrier = new_carrier
            new_carrier.add_order(self)
            print(f"Order {self.order_id} carrier changed to {new_carrier}.")