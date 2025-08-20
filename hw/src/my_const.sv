`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/16/2025 06:26:24 PM
// Design Name: 
// Module Name: my_const
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


package my_const;

    // constants used for the command registers
    localparam logic [7:0] CMD_IDLE       = 8'h00;
    localparam logic [7:0] CMD_CONVOLVE_0 = 8'h01;
    localparam logic [7:0] CMD_CONVOLVE_1 = 8'h02;
    localparam logic [7:0] CMD_POOL_0     = 8'h03;
    localparam logic [7:0] CMD_POOL_1     = 8'h04;
    localparam logic [7:0] CMD_DENSE_0    = 8'h05;
    localparam logic [7:0] CMD_DENSE_1    = 8'h06;
    localparam logic [7:0] CMD_DENSE_2    = 8'h07;
    
    // Status register constants
    localparam logic [31:0] STATUS_IDLE = 32'h0000;
    localparam logic [31:0] STATUS_BUSY = 32'h0001;
    localparam logic [31:0] STATUS_ERR  = 32'h0002;
    localparam logic [31:0] STATUS_DONE = 32'h0003;
    
    localparam N = 8;
    
    // ########################CONVOLTUON PARAMETERS########################
    localparam conv_n_0 = 28;
    localparam conv_n_1 = 14;
    
    localparam k = 5;
    localparam c_0 = 1;
    localparam c_1 = 6;
    
    // ########################POOLER PARAMETERS########################
    localparam pool_n_0 = 24;
    localparam pool_n_1 = 8;
    
    localparam s = 2;
    
    // ########################DENSE PARAMETERS########################
    localparam dense_n_0 = 256;
    localparam dense_n_1 = 120;
    localparam dense_n_2 = 84;
    
    localparam dense_m_0 = 120;
    localparam dense_m_1 = 84;
    localparam dense_m_2 = 10;
    

endpackage
