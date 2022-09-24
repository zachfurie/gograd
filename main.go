package main

import (
	"fmt"

	gograd_1d "github.com/zachfurie/gograd/gograd_1d"
	gograd_2d "github.com/zachfurie/gograd/gograd_2d"
)

func main() {
	fmt.Println("Hello World")
	gograd_1d.Run()
	fmt.Println("------------------")
	gograd_2d.Run()
}
