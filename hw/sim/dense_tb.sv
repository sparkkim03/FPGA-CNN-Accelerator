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

    parameter integer n = 6;
    parameter integer m = 4;
    parameter integer N = 16;
    parameter integer Q = 12;

    logic clk, rst, ce, val_dense_o;

    logic signed [N-1:0] data_i;
    logic signed [N-1:0] bias_i;
    logic signed [N-1:0] weight_i;
    
    logic signed [$clog2(n)-1:0] weights_x;
    logic signed [$clog2(m)-1:0] weights_y;
    
    logic signed [N-1:0] data_o;
    
    logic signed [N-1:0] weights [m-1:0][n-1:0];
    
    logic signed [N-1:0] output_buffer [m-1:0];
    
    integer i = 0;
    integer j = 0;

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
        .bias_i(bias_i),
        .weight_i(weight_i),
        .data_o(data_o),
        .val_dense_o(val_dense_o),
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
        weight_i <= 0;
        weights_x <= 0;
        weights_y <= 0;

        repeat(10) @(posedge clk);
        rst = 0;

        ce <= 1;
        
        repeat(1) @(posedge clk);
        
        ce <= 0;
        
        for(int i = 0; i < m; i++) begin
            for(int j = 0; j < n; j++) begin
                weights[i][j] <= i * n + j;
            end
        end
        
        // load the biases
        for(int i = 0; i < m; i++) begin
            bias_i <= i;
            @(posedge clk);
        end
        
        // load the data
        for(int i = 0; i < n; i++) begin
            data_i <= i;
            $display("Pushing data: %d",  data_i);
            @(posedge clk);
        end
        
        
        
        while (i < m) begin
            if (val_dense_o) begin
                @(posedge clk);
            end 
            else begin
                // If not stalled, update the weight and advance the counters.
                weight_i <= i * n + j;
                
                if (j == n - 1) begin
                    j = 0;
                    i = i + 1;
                end else begin
                    j = j + 1;
                end
                
                @(posedge clk);
            end
            
        end 
        
        
        repeat(1000) @(posedge clk);
        
        $finish();  
    end
    
    
 

endmodule
