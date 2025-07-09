`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/09/2025 04:09:38 PM
// Design Name: 
// Module Name: dense_tb
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


module dense_tb();
    
    timeunit 10ns;
    timeprecision 1ns;

    parameter integer n = 4;
    parameter integer m = 3;
    parameter integer N = 16;
    parameter integer Q = 12;

    logic clk, rst, ce;

    logic signed [N-1:0] data_i;

    dense #(
        .n(n),
        .m(m),
        .N(N),
        .Q(Q)
    ) dense_inst (
        .clk_i(clk),
        .rst_i(rst),
        .en_i(ce),
        .data_i(data_i),
        .bias_i({16'h0001, 16'h0002, 16'h0003}), // Example biases
        .data_o(),
        .val_dense_o(),
        .done_dense_o()
    );

    initial begin : CLK_INIT
        clk = 1;
    end

    always begin : CLK_GEN
        #1 clk = ~clk; 
    end 

    initial begin : TEST_VECTOR
        ce = 0;
        rst = 1;

        repeat(50) @(posedge clk);
        rst = 0;

        ce <= 1;
        
        repeat(1) @(posedge clk);
        
        ce <= 0;
        
        // Simulate input activations like the
        // Golden python script
        for(int i = 0; i < n; i++) begin
            data_i = i;
            //repeat(5) @(posedge clk);
            @(posedge clk);
            //ce <= 1;
        end
        
        
        repeat(1000) @(posedge clk);
        
        $finish();  
    end
 

endmodule
