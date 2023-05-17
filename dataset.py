import random
import pandas as pd

numbers = list(range(1,101)) * 5
labels = []

for num in numbers:
    rest_3 = num % 3
    rest_5 = num % 5

    if rest_3 == 0 and rest_5 == 0:
        labels.append("FizzBuzz")
    elif rest_3 == 0:
        labels.append("Fizz")
    elif rest_5 == 0:
        labels.append("Buzz")
    else:
        labels.append("None")

combined = list(zip(numbers, labels))
random.shuffle(combined)
randomized_numbers, randomized_labels = zip(*combined)

randomized_numbers = randomized_numbers[:500]
randomized_labels = randomized_labels[:500]

# Convert the number list into a 2D panda DataFrame
df = pd.DataFrame({'Number': numbers, 'Label': labels})
print(df)