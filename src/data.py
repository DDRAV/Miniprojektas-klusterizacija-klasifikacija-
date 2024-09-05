from db_engine import DBEngine
from carrier import Carrier
from manager import Manager
from customer import Customer
from warehouse import Warehouse
from order import Order


def load_carriers():
    """
   Fetches all carriers from the database and returns them as a dictionary of Carrier objects with key: client_id.
   :return: dictionary of carriers objects
   """
    db = DBEngine()
    query = "SELECT * FROM carrier"

    try:
        carrier_data = db.execute_sql(query)
        if not carrier_data:
            print("No carriers found in the database.")
            return {}

        carriers = {}  # Dictionary to store carriers

        for row in carrier_data:
            client_id, name, email, phone_number, country_of_reg = row
            carrier = Carrier(client_id, name, email, phone_number, country_of_reg)

            carriers[client_id] = carrier

        for client_id, carrier in carriers.items():
            print(f"Carrier ID {client_id}: {carrier}")

        return carriers



    except Exception as e:
        print(f"Error while loading carriers: {e}")
        return {}


def load_managers():
    """
   Fetches all managers from the database and returns them as a dictionary of Manager objects with key: manager_id.
   :return: dictionary of managers objects
   """
    db = DBEngine()
    query = "SELECT * FROM manager"

    try:
        manager_data = db.execute_sql(query)
        if not manager_data:
            print("No managers found in the database.")
            return {}

        managers = {}  # Dictionary to store carriers

        for row in manager_data:
            manager_id, name, surname, date_of_birth, email, phone_number, salary = row
            manager = Manager(manager_id, name, surname, date_of_birth, email, phone_number, salary)

            managers[manager_id] = manager

        for manager_id, manager in managers.items():
            print(f"Manager ID {manager_id}: {manager}")

        return managers

    except Exception as e:
        print(f"Error while loading carriers: {e}")
        return {}


def load_customers(managers):
    """
    Fetches all customers from the database and returns them as a dictionary of Customer objects with key: client_id.
    Each customer is associated with a manager based on manager_id.
    :param managers: dictionary of Manager objects
    :return: dictionary of Customer objects
    """
    db = DBEngine()
    query = "SELECT * FROM customer"

    try:
        customer_data = db.execute_sql(query)
        if not customer_data:
            print("No customers found in the database.")
            return {}

        customers = {}

        for row in customer_data:
            client_id, name, email, phone_number, status, manager_id = row

            manager = managers.get(manager_id)
            if not manager:
                print(f"Manager with ID {manager_id} not found for customer {name}.")
                continue

            customer = Customer(client_id, name, email, phone_number, status, manager)

            customers[client_id] = customer

        for customer_id, customer in customers.items():
            print(f"Customer ID {customer_id}: {customer}")

        return customers

    except Exception as e:
        print(f"Error while loading customers: {e}")
        return {}


def load_warehouses(customers):
    """
    Fetches all warehouses from the database and returns them as a dictionary of Warehouse objects with key: wh_id.
    Each warehouse is associated with a customer based on customer_id.
    :param customers: dictionary of Customer objects
    :return: dictionary of Warehouse objects
    """
    db = DBEngine()
    query = "SELECT * FROM warehouse"

    try:
        warehouse_data = db.execute_sql(query)
        if not warehouse_data:
            print("No warehouses found in the database.")
            return {}

        warehouses = {}

        for row in warehouse_data:
            wh_id, name, customer_id, country, postal_code, address, working_hours, wh_type = row

            customer = customers.get(customer_id)
            if not customer:
                print(f"Customer with ID {customer_id} not found for warehouse {name}.")
                continue

            warehouse = Warehouse(wh_id, name, customer, country, postal_code, address, working_hours, wh_type)

            warehouses[wh_id] = warehouse

        for wh_id, warehouse in warehouses.items():
            print(f"Warehouse ID {wh_id}: {warehouse}")

        return warehouses

    except Exception as e:
        print(f"Error while loading warehouses: {e}")
        return {}


def load_orders(warehouses, customers, managers, carriers):
    """
    Fetches all orders from the database and returns them as a list of Order objects.
    :param warehouses: dictionary of Warehouse objects
    :param customers: dictionary of Customer objects
    :param managers: dictionary of Manager objects
    :param carriers: dictionary of Carrier objects
    :return: list of Order objects
    """
    db = DBEngine()
    query = "SELECT * FROM orders"

    try:
        order_data = db.execute_sql(query)
        if not order_data:
            print("No orders found in the database.")
            return []

        orders = []

        for row in order_data:
            order_id, price, loading_wh_id, loading_time, unloading_wh_id, unloading_time, manager_id, customer_id, carrier_id = row

            loading_wh = warehouses.get(loading_wh_id)
            unloading_wh = warehouses.get(unloading_wh_id)
            manager = managers.get(manager_id)
            customer = customers.get(customer_id)
            carrier = carriers.get(carrier_id)

            print(f"Order ID: {order_id}, Loading Warehouse: {loading_wh}, Unloading Warehouse: {unloading_wh}")
            print(f"Manager: {manager}, Customer: {customer}, Carrier: {carrier}")

            if not (loading_wh and unloading_wh and manager and customer and carrier):
                print(f"Error: Missing data for order {order_id}. Skipping.")
                continue  # Skip this order if any object is missing

            order = Order(
                order_id=order_id,
                price=price,
                loading_wh=loading_wh,
                loading_time=loading_time,
                unloading_wh=unloading_wh,
                unloading_time=unloading_time,
                manager=manager,
                customer=customer,
                assembled_carrier=carrier
            )

            orders.append(order)

        return orders

    except Exception as e:
        print(f"Error while loading orders: {e}")
        return []


def load_data():
    carriers = load_carriers()
    managers = load_managers()
    customers = load_customers(managers)
    warehouses = load_warehouses(customers)
    orders = load_orders(warehouses, customers, managers, carriers)
    for order in orders:
        print(order.order_info())

    return carriers,managers,customers,warehouses,orders

