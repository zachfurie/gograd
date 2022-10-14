package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"time"

	// gograd_1d "github.com/zachfurie/gograd/gograd_1d"
	gograd_2d "github.com/zachfurie/gograd/gograd_2d"
)

func main() {
	numCPUs := runtime.NumCPU()
	runtime.GOMAXPROCS(numCPUs)
	fmt.Println(runtime.GOMAXPROCS(numCPUs))
	rand.Seed(time.Now().UnixNano())
	fmt.Println("------------------------------------------------------------------------")
	gograd_2d.Simple()
}
