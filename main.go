package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"time"

	// gograd_1d "github.com/zachfurie/gograd/gograd_1d"
	gograd_2d "github.com/zachfurie/gograd/gograd_2d"
	gograd_nd "github.com/zachfurie/gograd/gograd_nd"
)

func main() {
	d := 3
	numCPUs := runtime.NumCPU()
	runtime.GOMAXPROCS(numCPUs)
	fmt.Println("using ", runtime.GOMAXPROCS(numCPUs), " cores")
	rand.Seed(time.Now().UnixNano())
	fmt.Println("------------------------------------------------------------------------")
	if d == 2 {
		fmt.Println("2d")
		gograd_2d.Simple()
	} else if d == 3 {
		fmt.Println("Nd")
		gograd_nd.Simple()
		// gograd_nd.Test()
		// gograd_nd.TransposeTest()
	} else {
		fmt.Println("2d")
		gograd_2d.Simple()
		fmt.Println("------------------------------------------------------------------------")
		fmt.Println("Nd")
		gograd_nd.Simple()
	}
}
