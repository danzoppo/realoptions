// The R&D pharma model of Eduardo S Schwartz
// simulation real options model
package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// Assumptions about the R&D program

// CashProcess contains the assumptions of the cash flow simulation model.
type CashProcess struct {
	AnnualCashFlow     float64
	Drift              float64
	Volatility         float64
	TerminalMultiplier float64
	RiskPremium        float64
}

// Cash flow assumptions
const annualCashFlow float64 = 20e6
const cashDrift = 0.02
const cashVol = 0.35
const terminalCashMultiplier = 5
const riskPremium = 0.036
const adjCashDrift = cashDrift - riskPremium

// CostProcess is the assumptions of the cost and investment process.
type CostProcess struct {
	Investment        float64
	TotalExpectedCost float64
	Volatility        float64
	FailureProb       float64
}

// ProjectProcess contains the correlated structures of cost and cash flow processes.
type ProjectProcess struct {
	CashProcess
	CostProcess
	Correlation  float64
	RiskFreeRate float64
}

// Simulate returns the correlated cash and cost processes.
func (pp *ProjectProcess) Simulate() []*mat.Dense {
	// retVal is the simulated and correlated cost and cash flow processes for the project.
	retVal := make([]*mat.Dense, 2, 2)

	return retVal

}

// Cost assumptions
const annualInvestment = 10e6
const expectedCost = 100e6 // Expected total cost to completion
const costVol = 0.5
const costCashCorrel = -0.1

// Index assumptions
const timeStep = 0.25
const patentPeriod = 20                              // Years to patent expiration
const numberOfPeriods = int(patentPeriod / timeStep) // Number of periods in each run
const runs int = 10                                  // Number of simulations

// Market assumptions
const failureProb = 0.07
const riskFreeRate = 0.05

// functions
//
// Function importing and local alias
var exp = math.Exp
var sqrt = math.Sqrt
var stdNorm = rand.NormFloat64

// square returns the squared input value.
func sqr(x float64) float64 { return x * x }

// polynomial basis to approximate the value function
func basis(x, y float64) []float64 {
	return []float64{1, x, y, x * y, sqr(x), sqr(y), sqr(x) * y, x * sqr(y), sqr(x * y)}
}

func main() {

	// Initiate Cash Simulation
	netCash := mat.NewDense(runs, numberOfPeriods, nil)
	cost := mat.NewDense(runs, numberOfPeriods, nil)

	// stochastic simulation for cost to completion and net cash flow
	for run := 0; run < runs; run++ {
		for period := 0; period < numberOfPeriods; period++ {
			// correlate random variables
			costEps := stdNorm()
			cashEps := costCashCorrel*costEps + sqrt(1-sqr(costCashCorrel))*stdNorm()
			//fmt.Println(costEps)
			// cash simulation
			prevCash := annualCashFlow * timeStep
			if period != 0 {
				prevCash = netCash.At(run, period-1)
			}
			nextCash := prevCash * exp((adjCashDrift-0.5*sqr(cashVol))*timeStep+cashVol*sqrt(timeStep)*cashEps)
			netCash.Set(run, period, nextCash)

			// cost simulation
			prevCost := expectedCost
			if period != 0 {
				prevCost = cost.At(run, period-1)
			}

			if prevCost != 0 {
				prevCost = prevCost - annualInvestment*timeStep + costVol*sqrt(annualInvestment*prevCost*timeStep)*costEps
				if prevCost <= 0 {
					prevCost = 0
				}
				// if period == 0 {
				// 	fmt.Println(prevCost)
				// }
			}
			cost.Set(run, period, prevCost)
		}
	}

	// Value iteration
	valueArray := mat.NewDense(runs, numberOfPeriods, nil)

	// set terminal values
	const lastPeriod = numberOfPeriods - 1

	// Iterate over simulation paths and set terminal cash flow value
	for run := 0; run < runs; run++ {
		// If still investing in last period then value is zero otherwise set terminal CF multiple.
		if cost.At(run, lastPeriod) == 0 {
			termVal := terminalCashMultiplier * netCash.At(run, lastPeriod)
			valueArray.Set(run, lastPeriod, termVal)
		}
	}

	// Dsicount rates depending on phase of the project
	cashDiscRate := exp(-1 * riskFreeRate * timeStep)
	investDiscRate := exp(-1 * failureProb * timeStep)
	//fmt.Println(cashDiscRate, investDiscRate, investDiscRate*cashDiscRate)

	// Value function iteration
	for period := lastPeriod - 1; period >= 0; period-- {

		// discount next Value Function
		nextVal := mat.NewVecDense(runs, nil)
		nextVal.ScaleVec(cashDiscRate, valueArray.ColView(period+1))
		// if period == lastPeriod-1 {
		// 	fmt.Println(nextVal)
		// }

		// Basis matrix for regression
		BasisMat := mat.NewDense(runs, 9, nil)

		// still investing - multiply by investment discount rate
		for run := 0; run < runs; run++ {
			// set value function ot include discounting for investment
			if cost.At(run, period) != 0 {
				investVal := investDiscRate * nextVal.AtVec(run)
				nextVal.SetVec(run, investVal)

				// Value function approximation
			} else {
				nextVal.SetVec(run, nextVal.AtVec(run)+netCash.At(run, period))
			}

			// set basis matrix row
			BasisMat.SetRow(run, basis(cost.At(run, period), netCash.At(run, period)))

		}
		printAt(period, 0, nextVal)

		// solve for regression coefficients
		coefficients := mat.NewVecDense(9, nil)
		coefficients.SolveVec(BasisMat, nextVal)
		// if period == 0 {
		// 	fmt.Println(coefficients)
		// }

		estVal := mat.NewVecDense(runs, nil)
		estVal.MulVec(BasisMat, coefficients)

		for run := 0; run < runs; run++ {
			if cost.At(run, period) != 0 {
				periodVal := estVal.AtVec(run) - annualInvestment*timeStep
				if periodVal > 0 {
					valueArray.Set(run, period, periodVal)
				}
			} else {
				valueArray.Set(run, period, nextVal.AtVec(run))
			}

			//fmt.Println(periodVal)

		}

	}

	// Convert first month to slice of floats
	initialValue := make([]float64, runs)
	mat.Col(initialValue, 0, valueArray)
	//fmt.Println(initialValue)

	// Calculate average value of the initial period.
	retVal := stat.Mean(initialValue, nil)

	fmt.Printf("The Project Value: %.2f", retVal)

}

func printAt(period, when int, what interface{}) {
	if period == when {
		fmt.Println(what)
	}
}
