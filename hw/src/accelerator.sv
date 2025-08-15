`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/14/2025 03:01:19 PM
// Design Name: 
// Module Name: accelerator
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


module accelerator(
    input logic clk_i,
    input logic rst_i,
    
    // BRAM Access
    // BRAM_0 - DATA
    output logic [31:0]BRAM_PORTB_0_addr,
    output logic BRAM_PORTB_0_clk,
    output logic [31:0]BRAM_PORTB_0_din,
    input logic [31:0]BRAM_PORTB_0_dout,
    output logic BRAM_PORTB_0_en,
    output logic BRAM_PORTB_0_rst,
    output [3:0]BRAM_PORTB_0_we,
    
    // BRAM_1 - WEIGHT
    output logic [31:0]BRAM_PORTB_1_addr,
    output logic BRAM_PORTB_1_clk,
    output logic [31:0]BRAM_PORTB_1_din,
    input logic [31:0]BRAM_PORTB_1_dout,
    output logic BRAM_PORTB_1_en,
    output logic BRAM_PORTB_1_rst,
    output logic [3:0]BRAM_PORTB_1_we
    );
    
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
    
    // 2D convolve unit #1
    convolver #(
        .n(conv_n_0),
        .k(k),
        .c(c_0),
        .N(N)
    ) conv_28_28_1 (
    );
    
    // 2D convolve unit #2
    convolver #(
        .n(conv_n_1),
        .k(k),
        .c(c_1),
        .N(N)
    ) conv_14_14_6 (
    );
    
    // Max Pooling unit #1
    max_pooler #(
        .n(pool_n_0),
        .s(s),
        .N(N)
    ) pool_24_24 (
    );
    
    // Max Pooling unit #2
    max_pooler #(
        .n(pool_n_1),
        .s(s),
        .N(N)
    ) pool_8_8 (
    );
    
    // Dense unit #1
    dense #(
        .n(dense_n_0),
        .m(dense_m_0),
        .N(N)
    ) dense_256_120 (
    );
    
    // Dense unit #2
    dense #(
        .n(dense_n_1),
        .m(dense_m_1),
        .N(N)
    ) dense_120_84 (
    );
    
    // Dense unit #3
    dense #(
        .n(dense_n_2),
        .m(dense_m_2),
        .N(N)
    ) dense_84_10 (
    );
    
endmodule
