`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Engineer: Stefano Park Kim 
// 
// Create Date: 07/01/2025 12:49:25 PM
// Design Name: ReLU Module 
// Module Name:  ReLU
// Project Name: FPGA_CNN_Accelerator 
// Target Devices: Realdigital AUP-ZU3 
// Description: Performs ReLU on the input data
// 
// 
// Revision:
// Revision 0.01 - File Created 07/05/2025 11:49:25 PM
// Revision 0.02 - Initial Write up of the module completed 07/01/2025 14:58:00 PM
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module reLU # (
    parameter integer N = 16 // width of the data
)(
    input logic [N-1:0] data_i,
    output logic [N-1:0] data_o
);
    // check if the input data has its sign bit as 1
    assign data_o = (data_i[N-1]) ? 0 : data_i;

endmodule
