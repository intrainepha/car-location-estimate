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

	int: int data
*/
func Str2int(str string) int {
	intNum, err := strconv.Atoi(str)
	log.Println(err)
	return intNum
}

/*
Convert string to float64

Args:

	str(string): string data

Returns:

	intNum(float64): float64 data
*/
func Str2f64(str string) float64 {
	floatNum, err := strconv.ParseFloat(str, 64)
	log.Println(err)
	return floatNum
}

/*Convert float64 to string

Args:
	num(float64): float64 data

Returns:
	(string): string data
*/

func F642Str(num float64) string {
	return strconv.FormatFloat(num, 'g', -1, 64)
}
