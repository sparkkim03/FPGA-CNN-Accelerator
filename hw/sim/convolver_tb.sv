`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/01/2025 02:59:52 PM
// Design Name: 
// Module Name: convolver_tb
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


module convolver_tb();

    timeunit 10ns;
    timeprecision 1ns;

    parameter integer n = 5;    // Input matrix size (toy version)
    parameter integer k = 3;    // Kernel size (toy version)
    parameter integer N = 16;   // Total bit width
    parameter integer Q = 12;   // Fractional bit width

    logic clk, rst, ce;

    logic signed [N-1:0] activation;
    logic signed [N-1:0] conv_o;
    logic signed [N-1:0] weights;
    
    logic [N-1:0] bias;
    
    
    logic [N-1:0] kernel [0:k-1][0:k-1];

    logic val_conv_o, done_conv_o;
    
    int count;

    convolver #(
        .n(n),
        .k(k),
        .N(N),
        .Q(Q)
    ) conv (
        .clk_i(clk),
        .rst_i(rst),
        .en_i(ce),
        .feature_i(activation),
        .weight_i(weights),
        .bias_i(bias),
        .conv_o(conv_o),
        .val_conv_o(val_conv_o),
        .done_conv_o(done_conv_o)
    );

    initial begin : CLK_INIT
        clk = 1;
    end

    always begin : CLK_GEN
        #1 clk = ~clk; 
    end 
    
    initial begin: WEIGHTS_SETUP
        // Weights copied from the golden python script
        for(int i = 0; i < k; i++) begin
            for(int j = 0; j < k; j++) begin
                kernel[i][j] = count;
                count++;
            end
        end
    end

    initial begin : TEST_VECTOR
        ce = 0;
        rst = 1;
        activation = '0;
        count = 0;
        bias = 0;

        repeat(50) @(posedge clk);
        rst = 0;

        ce <= 1;
        
        @(posedge clk);
        
        ce <= 0;
        
        for(int x = 0; x < k; x++) begin
            for(int y = 0; y < k; y++) begin
                weights = kernel[x][y];
                @(posedge clk);
            end
        end
        
        // Simulate input activations like the
        // Golden python script
        for(int i = 0; i < n*n; i++) begin
            activation = i;
            @(posedge clk);
        end
        
        
        repeat(1000) @(posedge clk);
        
        $finish();  
    end
endmodule
