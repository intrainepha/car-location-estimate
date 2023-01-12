package ops

import (
	"log"
	"strconv"
)

/*
Convert string to int

Args:

	string: string data

Returns:

	iNum: int data
*/
func Stoi(str string) int {
	iNum, err := strconv.Atoi(str)
	if err != nil {
		log.Panic(err)
	}
	return iNum
}

/*
Convert string to float64

Args:

	str string: string data

Returns:

	fNum float64: float64 data
*/
func Stof(str string) float64 {
	fNum, err := strconv.ParseFloat(str, 64)
	if err != nil {
		log.Panic(err)
	}
	return fNum
}

/*
Convert float64 to string

Args:

	num float64: float64 data

Returns:

	string: string data
*/
func Ftos(num float64) string {
	return strconv.FormatFloat(num, 'g', -1, 64)
}
