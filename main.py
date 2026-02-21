from models_rework import (
    file_check,
    print_data_overview,
    user_choice,
    evaluate_model,
    custom_predict
)


def menu():

    print("-" * 50)
    print("1. Import data")
    print("2. Train model")
    print("3. Evaluate model")
    print("4. Simulate new input")
    print("5. Exit")
    print("-" * 50)

    while True:
        choice = input("Choose (1-5): ").strip()
        if choice in ["1", "2", "3", "4", "5"]:
            return choice
        print("Invalid input. Please enter 1-5.")

def main():

    df = None
    model = None

    while True:

        choice = menu()

        if choice == "1":
            df = file_check()
            if df is not None:
                print_data_overview(df)
            input("Press ENTER to continue...")

        elif choice == "2":
            if df is None:
                print("Load data first (option 1).")
            else:
                model = user_choice(df)
            input("Press ENTER to continue...")

        elif choice == "3":
            if df is None:
                print("Load data first (option 1).")
            else:
                evaluate_model(df)
            input("Press ENTER to continue...")

        elif choice == "4":
            if df is None:
                print("Load data first (option 1).")
            else:
                custom_predict(df, model)
            input("Press ENTER to continue...")

        elif choice == "5":
            print("Exiting program.")
            break


if __name__ == "__main__":
    main()