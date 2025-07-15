`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/15/2025 10:25:35 AM
// Design Name: 
// Module Name: bias_reg
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


module bias_reg #(
    parameter integer N = 16,
    parameter integer Q = 12
)(
    input logic [6:0] addr_i,
    input logic [2:0] sel_i,
    output logic [N-1:0] bias_o
    );
    
    
    logic [N-1:0] conv1_bias [5:0]; // 6 baises
    logic [N-1:0] conv2_bias [15:0]; // 16 biases
    logic [N-1:0] fc1_bias [119:0]; // 120 biases
    logic [N-1:0] fc2_bias [83:0]; // 84 biases
    logic [N-1:0] fc3_bias [9:0]; // 10 biases
    
    assign conv1_bias = {16'hfe18, 16'hfc08, 16'hffa4, 16'hffd5, 16'h0133, 16'h0277};
    assign conv2_bias = {16'h01f6, 16'h0000, 16'h0151, 16'h00c8, 16'h00bd, 16'h015b, 
                   16'hfe72, 16'hfee8, 16'hff7c, 16'hffcd, 16'hffef, 16'hfff2, 
                   16'hfe7f, 16'h0037, 16'hfe11, 16'h0036};
    assign fc1_bias = {16'hfffb, 16'hffa2, 16'h000c, 16'hff54, 16'hfdf7, 16'h0187,
                   16'h0130, 16'h005b, 16'hff37, 16'hffd6, 16'hffb4, 16'hfff7,
                   16'h001a, 16'h0164, 16'hfeb5, 16'hfe86, 16'hffe3, 16'hff4c,
                   16'hff78, 16'h0055, 16'hffb1, 16'hfff5, 16'h0055, 16'h0059,
                   16'h0002, 16'hfe8c, 16'hff85, 16'hff05, 16'hfed9, 16'h017b,
                   16'hff21, 16'hff5b, 16'h00f2, 16'h0001, 16'h007e, 16'h0097,
                   16'h00f1, 16'h017f, 16'hffa3, 16'hff6c, 16'hfdca, 16'hffbb,
                   16'hfed8, 16'hffa5, 16'h00f4, 16'h00d3, 16'h01d4, 16'h007d,
                   16'hff67, 16'hfef4, 16'hff90, 16'hff16, 16'hffa4, 16'h006a,
                   16'h0090, 16'h0082, 16'h0021, 16'hfef6, 16'h009d, 16'hffd6,
                   16'h016c, 16'hff63, 16'hffb0, 16'h00f5, 16'hff30, 16'h00bb,
                   16'hffe2, 16'hfe8e, 16'hffbd, 16'hffa8, 16'hff5b, 16'h004e,
                   16'h01ea, 16'h0166, 16'hffcd, 16'h0089, 16'hff87, 16'h0011,
                   16'h0085, 16'h0012, 16'h0020, 16'h002b, 16'hff08, 16'hffd7,
                   16'h0067, 16'h0031, 16'hffc8, 16'hff4d, 16'hff4c, 16'hfea6,
                   16'hff7c, 16'hfe5e, 16'h0085, 16'hff69, 16'hff32, 16'h0068,
                   16'h0037, 16'h007e, 16'hff04, 16'hfe6d, 16'h01d6, 16'h00a1,
                   16'h0071, 16'hff47, 16'hffc1, 16'hffb8, 16'h006d, 16'h0081,
                   16'h005b, 16'h0020, 16'h0055, 16'hff6d, 16'h00f5, 16'h0064,
                   16'hffd9, 16'hffbb, 16'hff04, 16'h017c, 16'hff30, 16'h0146};
    assign fc2_bias = {16'h0181, 16'h0178, 16'h028b, 16'h0002, 16'hff27, 16'h01d7,
                   16'hfefd, 16'h001f, 16'h01fb, 16'hff5a, 16'h000e, 16'h00bb,
                   16'hfea0, 16'h004b, 16'h00f0, 16'h0219, 16'h00b4, 16'hff6d,
                   16'hff56, 16'hff13, 16'hff40, 16'h014d, 16'h00a3, 16'hfda1,
                   16'hffcd, 16'h0203, 16'hff5d, 16'h0107, 16'h0034, 16'h0269,
                   16'hffff, 16'h0025, 16'hfeda, 16'h0065, 16'h00c0, 16'h0006,
                   16'hffdd, 16'h020c, 16'h003f, 16'h0049, 16'h0101, 16'h0204,
                   16'hfd83, 16'h0116, 16'h01af, 16'hfe7c, 16'hffe7, 16'hffcc,
                   16'h0183, 16'hfed1, 16'hffe6, 16'h00c6, 16'h0065, 16'h0179,
                   16'hfe62, 16'hfe6c, 16'hff2b, 16'hfd9b, 16'hff57, 16'h00d4,
                   16'h0047, 16'h01e3, 16'hff66, 16'h01a1, 16'h0099, 16'h01cf,
                   16'hff98, 16'h00cc, 16'hfda7, 16'h00fe, 16'h0169, 16'h017a,
                   16'h0197, 16'h0000, 16'h014c, 16'h01b3, 16'hfdd9, 16'h001e,
                   16'hfed9, 16'hff0c, 16'h0081, 16'hff33, 16'hfe4f, 16'hfeb8};
    assign fc3_bias = {16'hfeff, 16'h01db, 16'hfdd0, 16'hff2e, 16'h00f6, 16'hff2d,
                   16'hff4f, 16'hff8b, 16'h002f, 16'hffd2};
                   
    always_comb begin
        bias_o = 0;
        case(sel_i)
            3'b000: begin
                bias_o = conv1_bias[addr_i];
            end
            3'b001: begin
                bias_o = conv2_bias[addr_i];
            end
            3'b010: begin
                bias_o = fc1_bias[addr_i];
            end
            3'b011: begin
                bias_o = fc2_bias[addr_i];
            end
            3'b100: begin
                bias_o = fc3_bias[addr_i];
            end
        endcase
    end
    
endmodule
