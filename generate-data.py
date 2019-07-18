import numpy as np
import cx_Oracle

db = cx_Oracle.connect(user="ADMIN", password="Oracle12345!", dsn="mlwadw_high")
print("Connected to Oracle ADW")

def insertData(db, x_set, y_val):
    cursor = db.cursor()
    cursor.execute("""
            INSERT INTO SAMPLE_DATA 
                (X1, X2, X3, X4, X5, X6, X7 ,X8, X9, X10, Y)
            VALUES
                (:x1, :x2, :x3, :x4, :x5, :x6, :x7, :x8, :x9, :x10, :y)  
        """,
        x1 = x_set[0],
        x2 = x_set[1],
        x3 = x_set[2],
        x4 = x_set[3],
        x5 = x_set[4],
        x6 = x_set[5],
        x7 = x_set[6],
        x8 = x_set[7],
        x9 = x_set[8],
        x10 = x_set[9],
        y = y_val
    )
    db.commit()
            
print("Setting up data generation...")
np.random.seed(12)
num_observations = 5
num_rows = 5000000
# multivariate_normal returns 2 values, which are flattened to make 10 columns
fieldnames = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'y']

print("Begin generating data...")
for x in range (num_rows):
    set1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
    set1 = set1.flatten()
    insertData(db, set1, 0)
    

    set2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)
    set2 = set2.flatten()
    insertData(db, set2, 1)
    pctComplete = (x / num_rows) * 100
    print ("{:.2f}".format(pctComplete)+"%", end="\r")

print("Finished generating data...")
