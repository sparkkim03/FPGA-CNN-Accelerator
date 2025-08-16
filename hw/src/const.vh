// constants used for the command registers
`define CMD_IDLE      8'h00
`define CMD_CONVOLVE  8'h01
`define CMD_POOL      8'h02
`define CMD_DENSE     8'h03

// constants used for the status registers
`define STATUS_IDLE     8'h00
`define STATUS_BUSY     8'h01
`define STATUS_ERR      8'h02
`define STATUS_DONE     8'h03