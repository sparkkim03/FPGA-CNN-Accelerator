`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: Stefano Park Kim 
// 
// Create Date: 07/09/2025 03:28:32 PM
// Design Name: Dense Layer Module 
// Module Name: dense
// Project Name: FPGA_CNN_Accelerator 
// Target Devices: AUP-ZU3 
// Description: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module dense #(
    parameter integer n = 256, // number of inputs
    parameter integer m = 120,   // number of outputs
    parameter integer N = 16,   // total bit width
    parameter integer Q = 12    // fractional bit width
)(
    input logic clk_i,
    input logic rst_i,
    input logic en_i,

    input logic signed [N-1:0] data_i, // input data
    input logic signed [N-1:0] bias_i[m-1:0], // bias for the dense layer

    output logic signed [N-1:0] data_o, // output data
    output logic val_dense_o, // valid signal for the output data
    output logic done_dense_o // done signal for the dense layer
);
    logic wea;
    // both address width fixed to largest instances (input 256 output 120)
    logic [7:0] input_addr; 
    logic [14:0] weight_addr;  
   
    logic signed [N-1:0] input_reg;
    logic signed [N-1:0] weight_reg;
    
    logic [$clog2(n)-1:0] input_counter; // number of inputs, multiplexed to input_addr
    
    logic [$clog2(n)-1:0] dotproduct_coutner; // inner loop counter, multiplexed to input_addr
    logic [$clog2(m)-1:0] neuron_counter; // outer loop counter
    
    logic signed [N*2-1:0] accum_reg;
    
    // the weight address will just be calculated like this
    // changed by changing neuron_counter and input_address values
    assign weight_addr = neuron_counter * m + input_addr;
    
    // both bram created assuming max size
    dense_input i_mem(
        .addra(addr_m),
        .clka(clk_i),
        .dina(data_i),
        .douta(input_reg),
        .wea(wea)
    );
    // data for weights will be preloaded, no need to write
    dense_weights w_mem(
        .addra(),
        .clka(clk_i),
        .douta(weight_reg)
    );
    
    enum logic [1:0] {
        IDLE,
        LOAD,
        PROCESSING,
        DONE
    } state, next_state;  

    always_comb begin
        wea = 0;
        val_dense_o = 0;
        done_dense_o = 0;

        case(state)
            IDLE: begin
                wea = 0;
            end
            
            LOAD: begin
                wea = 1; // write enable for loading data
            end
            
            PROCESSING: begin
                wea = 0; // no writing during processing
            end
            
            DONE: begin
                wea = 0; // no writing during done state
            end
            
            default: begin
                wea = 0;
            end
        endcase
    end

    always_comb begin
        next_state = state;
        case(state)
            IDLE: begin
                if(en_i) begin
                    next_state = LOAD;
                end
            end
            
            LOAD: begin
                if(addr_m == n - 1) begin
                    next_state = PROCESSING;
                end
            end
            
            PROCESSING: begin 
            end
            
            DONE: begin
                if(~en_i) begin
                    next_state = IDLE;
                end else begin
                    next_state = DONE;
                end
            end
            
            default: next_state = IDLE;
        endcase
    end 
    
    always_ff @(posedge clk_i or posedge rst_i) begin
        if (rst_i) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end
    
endmodule
