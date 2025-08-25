using CSV, DataFrames, MLUtils
using BenchmarkTools
using Profile

# Dataset loading
df = CSV.read("./data/mnist.csv", DataFrame)
X = transpose(Matrix(df[:, 1:end-1]))
y = Vector(df[:, end])

println("X size: ", size(X))
println("y length: ", length(y))

# Split train/test (80% - 20%)
train, test = splitobs((X, y), at = 0.8)

X_train, y_train = train
X_test, y_test = test;

d = size(X)[1]
m = 50_000
k = floor(Int64, d*log(ℯ,m)) # k = d*ln(m)
seed = 42

model = EaSClassifier.fit(X_train, y_train, m, k, seed)
scores = EaSClassifier.predict(X_test, model)

y_pred = Int.(scores .>= 0.5);

@btime model = EaSClassifier.fit(X_train, y_train, m, k, seed)
