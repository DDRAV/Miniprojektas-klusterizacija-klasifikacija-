from db_engine import DBEngine
class Manager:
    def __init__(self, manager_id: int, name: str, surname: str, date_of_birth: int, email: str, phone_number: int, salary: int):
        self.manager_id = manager_id
        self.name = name
        self.surname = surname
        self.date_of_birth = date_of_birth
        self.email = email
        self.phone_number = phone_number
        self.salary = salary
        self.manager_customers = []
        self.manager_orders = []


    def __repr__(self):
        return (f"Manager {self.manager_id}, {self.name}, {self.surname}, {self.phone_number}")

    def manager_info(self):
        print(f"Manager ID: {self.manager_id}\n"
            f"Name: {self.name}\n"  
            f"Surname: {self.surname}\n"
            f"Date of birth: {self.date_of_birth}\n"
            f"Email: {self.email}\n"
            f"Phone number: {self.phone_number}\n"
            f"Salary: {self.salary}\n")

    def change_manager_info(self, manager_id=None, name=None, surname=None, date_of_birth=None, email=None, phone_number=None, salary=None):
        if manager_id:
            self.manager_id = manager_id
        if name:
            self.name = name
        if surname:
            self.surname = surname
        if date_of_birth:
            self.date_of_birth = date_of_birth
        if email:
            self.email = email
        if phone_number:
            self.phone_number = phone_number
        if salary:
            self.salary = salary

    def add_customer(self, customer):
        if customer not in self.manager_customers:
            self.manager_customers.append(customer)
            print (f"Customer {customer} added for manager {self.__repr__()}")
        else:
            print(f"Customer {customer} already is in manager{self.__repr__()} customers list")


    def show_manager_customers(self):
        print(f"Manager {self.__repr__()} is managing customers:\n"
              f"{self.manager_customers}")

    def add_order(self, order):
        if order not in self.manager_orders:
            self.manager_orders.append(order)
            print(f"Order {order} added for manager {self.__repr__()}")
        else:
            print(f"Order {order} already is in manager{self.__repr__()} order list")

    def show_manager_orders(self):
        print(f"Manager {self.__repr__()} is managing orders:\n"
              f"{self.manager_orders}")

