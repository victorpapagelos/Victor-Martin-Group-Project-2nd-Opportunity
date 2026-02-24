import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# Load file
def file_check():
    while True:
        fn = input("Enter CSV file (or q to cancel): ").strip()
        if fn.lower() == "q":
            return None
        try:
            df = pd.read_csv(fn)
            print("File loaded.")
            return df
        except FileNotFoundError:
            print("File not found.")

# Yes / No
def yes_no_prompt(prompt="(y/n): "):
    while True:
        r = input(prompt).strip().lower()
        if r == "y": return True
        if r == "n": return False
        print("Enter y or n.")


# Test size
def get_test_size():
    while True:
        s = input("Test size (0.1-0.9): ").replace(",", ".")
        try:
            v = float(s)
            if 0.1 <= v <= 0.9:
                return v
        except:
            pass
        print("Invalid number.")

# Show data
def print_data_overview(df):
    print("Top 10 rows:")
    print(df.head(10))
    print("\nStatistics:")
    print(df.describe(include="all")) #column types


# mapping P/A/N
mapping = {"N": 0, "A": 1, "P": 2}

def map_data(X):
    X = X.copy()
    for col in X.columns:
        X[col] = X[col].map(mapping)
    return X


# Train Decision Tree
def decision_tree_model(df):

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X = map_data(X)

    test_size = get_test_size()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)

    preds = dt.predict(X_test)

    print("Decision Tree")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    return dt


# Train kNN
def knn_model(df):

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X = map_data(X)

    test_size = get_test_size()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    preds = knn.predict(X_test)

    print("kNN")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    return knn

# Model selection
def user_choice(df):

    if df is None:
        print("Load data first.")
        return None

    print("1. Decision Tree")
    print("2. kNN")
    c = input("Choose: ").strip()

    if c == "1":
        return decision_tree_model(df)
    elif c == "2":
        return knn_model(df)
    else:
        print("Invalid choice.")
        return None
    

# Evaluate
def evaluate_model(df):

    if df is None:
        print("Load data first.")
        return

    print("1. Decision Tree")
    print("2. kNN")
    print("3. Both")

    while True:
        choice = input("Choose (1-3): ").strip()
        if choice in ["1", "2", "3"]:
            break
        print("Invalid input.")
    
    use_external = yes_no_prompt("Do you want to load an external evaluation file? (y/n): ")

    results = []

    if use_external:

        eval_df = file_check()
        if eval_df is None:
            print("Failed to load evaluation file.")
            return
        if "Class" not in eval_df.columns:
            print("No 'Class' column found.")
            return

        # Train on FULL df
        X_train = map_data(df.drop("Class", axis=1))
        y_train = df["Class"]

        # Test on external file
        X_test = map_data(eval_df.drop("Class", axis=1))
        y_test = eval_df["Class"]

    else:
        # Split df
        X = map_data(df.drop("Class", axis=1))
        y = df["Class"]

        test_size = get_test_size()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

    def run(model, name):
        model.fit(X_train, y_train)
        p = model.predict(X_test)
        acc = accuracy_score(y_test, p)
        report = classification_report(y_test, p)

        print(f"\n{name}")
        print("Accuracy:", acc)
        print(report)

        results.append((name, acc, report))

    if choice == "1":
        run(DecisionTreeClassifier(), "Decision Tree")
    elif choice == "2":
        run(KNeighborsClassifier(n_neighbors=1), "kNN")
    else:
        run(DecisionTreeClassifier(), "Decision Tree")
        run(KNeighborsClassifier(n_neighbors=1), "kNN")

    # Save option
    if yes_no_prompt("Save results to file? (y/n): "):

        filename = input("Enter file name (default: results.txt): ").strip()

        if filename == "":
            filename = "results.txt"

        if not filename.endswith(".txt"):
            filename += ".txt"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                for name, acc, report in results:
                    f.write(f"{name}\n")
                    f.write(f"Accuracy: {acc}\n")
                    f.write(report)
                    f.write("\n\n")
            print("Results saved.")
        except:
            print("Error saving file.")



# Create custom input
def custom_input():

    print("\nPlease Select Values For Each Attribute In The Custom Scenario.")
    print("P = Positive | A = Average | N = Negative\n")

    def get_val(text):
        while True:
            v = input(text).upper()
            if v in ("P", "A", "N"):
                return v
            print("Use P, A or N.")

    return pd.DataFrame({
        "Industrial_Risk": [get_val("Industrial Risk (P/A/N): ")],
        "Management_Risk": [get_val("Management Risk P/A/N: ")],
        "Financial_Flexibility": [get_val("Financial Flexibility P/A/N: ")],
        "Credibility": [get_val("Credibility P/A/N: ")],
        "Competitiveness": [get_val("Competitiveness P/A/N: ")],
        "Operating_Risk": [get_val("Operating Risk P/A/N: ")],
    })

# Predict custom
def custom_predict(df, model):

    if df is None:
        print("Load data first.")
        return

    if model is None:
        print("Train model first.")
        return

    new_row = custom_input()
    new_row = map_data(new_row)

    pred = model.predict(new_row)

    print("\nPredicted Class:", pred[0])