numbers = [
     14, 25, 21, 14, 18, 19, 20, 23, 15, 17, 13, 18, 20, 26, 13, 13, 16, 19,
     11, 20, 25, 19, 12, 26, 23, 20, 13, 15, 17, 22, 14, 22, 20, 17, 11, 13,
     24, 16, 23, 20, 18, 19, 18, 21, 9, 13, 16, 19, 20, 14, 14
]

print("Minimum:", min(numbers))

print("Maximum:", max(numbers))

print("Count:", len(numbers))

count_greater_than_20 = sum(1 for n in numbers if n > 20)
print("Count of numbers > 20:", count_greater_than_20)

sorted_numbers = sorted(numbers)
print("Sorted Numbers:", sorted_numbers)
