`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/05/2025 01:36:00 PM
// Design Name: 
// Module Name: max_pooler_tb
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


module max_pooler_tb();

    timeunit 10ns;
    timeprecision 1ns;

    parameter integer n = 8;    // Input matrix size (toy version)
    parameter integer s = 2;    // pool size 
    parameter integer N = 16;   // Total bit width
    parameter integer Q = 12;   // Fractional bit width
    
    logic clk, rst, ce;
    
    logic [N-1:0] input_r, output_r;
    
    logic val_pool, done_pool;
    
    max_pooler #(
        .n(n),
        .s(s),
        .N(N)
    )pool(
        .clk_i(clk),
        .rst_i(rst),
        .en_i(ce),
        .data_i(input_r),
        .data_o(output_r),
        .val_pool_o(val_pool),
        .done_pool_o(done_pool)
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
        input_r = '0;

        repeat(50) @(posedge clk);
        rst = 0;

        ce <= 1;
        
        repeat(2) @(posedge clk);
        
        ce <= 0;
        
        // Simulate input activations like the
        // Golden python script
        for(int i = 0; i < n*n; i++) begin
            input_r = i;
            //repeat(5) @(posedge clk);
            @(posedge clk);
            //ce <= 1;
        end
        
        
        repeat(1000) @(posedge clk);
        
        $finish();  
    end
    
endmodule
