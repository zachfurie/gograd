package main

import (
	"fmt"
	"math/rand"
	"time"

	// gograd_1d "github.com/zachfurie/gograd/gograd_1d"
	gograd_2d "github.com/zachfurie/gograd/gograd_2d"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	fmt.Println("------------------------------------------------------------------------")
	gograd_2d.Run()
}
