// Package main runs the R&D Real Option Model
package main

import (
	"fmt"
	"math"

	"github.com/leekchan/accounting"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// CashProcess contains the assumptions of the cash flow simulation
type CashProcess struct {
	AnnualCashFlow     float64
	Drift              float64
	Volatility         float64
	TerminalMultiplier float64
	RiskPremium        float64
}

// CostProcess is the assumptions of the cost and investment process
type CostProcess struct {
	Investment        float64
	TotalExpectedCost float64
	Volatility        float64
	FailureProb       float64
}

// Simulation holds the assumptions for the Monte Carlo simulation.
type Simulation struct {
	TimeStep        float64
	PatentLength    int
	Runs            int
	PolynomialOrder int
}

// ProjectProcess contains the correlated structures of the cost
// and cash flow processes comprising the project.
// Correlation correlates the CashProcess and the CostProcess.
type ProjectProcess struct {
	CashProcess
	CostProcess
	Correlation  float64
	RiskFreeRate float64
	Simulation
}

func main() {
	// Set random seed
	rand.Seed(355)

	// Initialize project components
	cashProcess := CashProcess{
		AnnualCashFlow:     20e6,
		Drift:              0.02,
		Volatility:         0.35,
		TerminalMultiplier: 5,
		RiskPremium:        0.036,
	}
	investmentProcess := CostProcess{
		Investment:        10e6,
		TotalExpectedCost: 100e6,
		Volatility:        0.5,
		FailureProb:       0.06931,
	}
	sim := Simulation{
		TimeStep:        0.25,
		PatentLength:    20,
		Runs:            200_000,
		PolynomialOrder: 9,
	}

	project := ProjectProcess{
		CashProcess:  cashProcess,
		CostProcess:  investmentProcess,
		Correlation:  -0.1,
		RiskFreeRate: 0.05,
		Simulation:   sim,
	}

	// Estimate Project Value
	projectValue := project.Lsm()

	// Print currency
	ac := accounting.Accounting{Symbol: "$", Precision: 2}
	fmt.Println("The Project Value:", ac.FormatMoney(projectValue))
}

// Simulate returns the correlated cash and cost processes.
func (pp *ProjectProcess) Simulate() (*mat.Dense, *mat.Dense) {

	// Set number of periods
	numberOfPeriods := int(float64(pp.PatentLength) / pp.TimeStep)

	// Risk adjusted cash flow drift rate
	adjCashDrift := pp.CashProcess.Drift - pp.RiskPremium

	// Matrices to hold the simulated cash and cost values
	netCash := mat.NewDense(pp.Runs, numberOfPeriods, nil)
	cost := mat.NewDense(pp.Runs, numberOfPeriods, nil)

	// stochastic simulation of the investment costs and cash flows
	for run := 0; run < pp.Runs; run++ {
		for period := 0; period < numberOfPeriods; period++ {

			// correlate random variables
			costEps := rand.NormFloat64()
			cashEps := pp.Correlation*costEps + sqrt(1-sqr(pp.Correlation))*rand.NormFloat64()

			// cash flow simulation
			prevCash := pp.AnnualCashFlow
			if period != 0 {
				prevCash = netCash.At(run, period-1)
			}
			nextCash := prevCash * exp((adjCashDrift-0.5*sqr(pp.CashProcess.Volatility))*pp.TimeStep+
				pp.CashProcess.Volatility*sqrt(pp.TimeStep)*cashEps)
			netCash.Set(run, period, nextCash)

			// cost simulation
			prevCost := pp.TotalExpectedCost
			if period != 0 {
				prevCost = cost.At(run, period-1)
			}

			// Only update costs if not zero
			nextCost := 0.0
			if prevCost != 0 {
				nextCost = prevCost - pp.Investment*pp.TimeStep +
					pp.CostProcess.Volatility*sqrt(pp.Investment*prevCost*pp.TimeStep)*costEps
				if nextCost < 0 {
					nextCost = 0
				}
			}
			cost.Set(run, period, nextCost)

		}

	}

	return netCash, cost
}

// Lsm evaluates the project using the Least Squares Monte Carlo algorithm
func (pp *ProjectProcess) Lsm() float64 {
	// Calc periods
	numberOfPeriods := int(float64(pp.PatentLength) / pp.TimeStep)
	lastPeriod := numberOfPeriods - 1

	// Simulate the cost and cash flow values
	cashMatrix, costMatrix := pp.Simulate()

	// valueArray holds the value function iteration matrix
	valueArray := mat.NewDense(pp.Runs, numberOfPeriods, nil)

	// Set the Terminal Value
	for run := 0; run < pp.Runs; run++ {
		// If cost is positive then still investing
		// and no value in terminal period.
		if costMatrix.At(run, lastPeriod) == 0 {
			termVal := pp.TerminalMultiplier * cashMatrix.At(run, lastPeriod)
			valueArray.Set(run, lastPeriod, termVal)
		}

		// if costMatrix.At(run, lastPeriod) != 0 {
		// 	fmt.Println(valueArray.At(run, lastPeriod))
		// }
	}

	// Discount rates depending on the phase of the project
	cashDiscRate := exp(-1 * pp.RiskFreeRate * pp.TimeStep)
	investDiscRate := exp(-1 * (pp.RiskFreeRate + pp.FailureProb) * pp.TimeStep)

	// Value iteration
	for period := lastPeriod - 1; period >= 0; period-- {

		// Initialize next periods Value
		nextVal := mat.NewVecDense(pp.Runs, nil)

		// Discount next period's value to serve as the dependent variable of the regression
		nextVal.ScaleVec(investDiscRate, valueArray.ColView(period+1))

		// Initialize basis matrix for regression
		basisMatrix := mat.NewDense(pp.Runs, 9, nil)

		// Set basis matrix rows for the regression
		for run := 0; run < pp.Runs; run++ {
			basisMatrix.SetRow(run, basis(costMatrix.At(run, period), cashMatrix.At(run, period)))
		}

		// Solve for regression coefficients
		coefficients := mat.NewVecDense(9, nil)
		coefficients.SolveVec(basisMatrix, nextVal)

		// Estimate continuation value of investment
		estVal := mat.NewVecDense(pp.Runs, nil)
		estVal.MulVec(basisMatrix, coefficients)

		// Determine Value and set in valueArray
		for run := 0; run < pp.Runs; run++ {
			// Investing Value
			if costMatrix.At(run, period) != 0 {
				investVal := estVal.AtVec(run) - pp.Investment*pp.TimeStep
				// Only invest if project value is positive after investment
				if investVal > 0 {
					valueArray.Set(run, period, nextVal.AtVec(run)-pp.Investment*pp.TimeStep)
				}
			} else {
				// Post investment sales cash flow
				valueArray.Set(run, period, cashMatrix.At(run, period)*pp.TimeStep+
					cashDiscRate*valueArray.At(run, period+1))
			}
		}
	}

	// Convert first month to slice of floats
	initialValue := make([]float64, pp.Runs)
	mat.Col(initialValue, 0, valueArray)

	// Discount one last time to initial period
	floats.ScaleTo(initialValue, cashDiscRate*investDiscRate, initialValue)

	// Average discounted initial period across all runs
	retVal := stat.Mean(initialValue, nil)

	return retVal
}

// helper functions

// square the input
func sqr(x float64) float64 { return x * x }

// polynomial basis to approximate the value function
func basis(x, y float64) []float64 {
	return []float64{1, x, y, x * y, sqr(x), sqr(y), sqr(x) * y, x * sqr(y), sqr(x * y)}
}

// local function aliases
var exp = math.Exp
var sqrt = math.Sqrt
