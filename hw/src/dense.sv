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
    logic [7:0] input_addr_write;
    logic [7:0] input_addr_read; 
    logic [14:0] weight_addr;  
    
    logic signed [N-1:0] input_data;
    logic signed [N-1:0] weight_data;
    
    logic [$clog2(n)-1:0] input_counter; // number of inputs, multiplexed to input_addr
    
    logic [$clog2(n):0] x_counter; // inner loop counter, multiplexed to input_addr
    logic [$clog2(m)-1:0] y_counter; // outer loop counter
    
    logic signed [N-1:0] accumulator;
    logic signed [N-1:0] data_temp;
    logic signed [N*2-1:0] product;

    // both bram created assuming max size
    dense_input i_mem(
        .addra(input_addr_write),
        .clka(clk_i),
        .dina(data_i),
        .wea(wea),
        .addrb(input_addr_read),
        .clkb(clk_i),
        .doutb(input_data),
        .enb(1'b1)
    );
    // data for weights will be preloaded, no need to write
    dense_weights w_mem(
        .addra(weight_addr),
        .clka(clk_i),
        .douta(weight_data),
        .wea(1'b0) 
    );
    
    enum logic [1:0] {
        IDLE,
        LOAD,
        PROCESSING,
        STORE
    } state, next_state;  
    
    always_comb begin
        // the weight address will just be calculated like this
        // changed by changing neuron_counter and input_address values
        weight_addr = x_counter * m + y_counter; // n*m weights
        //weight_addr = y_counter * n + x_counter;
        
        input_addr_write = input_counter;
        input_addr_read = x_counter;
        
        product = input_data * weight_data;
    end
    
    always_comb begin
        data_o = 0;
        val_dense_o = 0;
        done_dense_o = 0;
        wea = 0;
        
        case(state) 
            LOAD: begin
                wea = 1;
            end
            STORE: begin
                val_dense_o = 1;
                // applying bias and relu
                data_temp = accumulator + bias_i[y_counter];
                data_o = (data_temp > 0) ? data_temp : 0;
                
                if(y_counter == m - 1) begin
                    done_dense_o = 1;
                end
            end
        endcase
    end
    
    always_comb begin
        next_state = state;
        case(state)
            IDLE: begin
                if(en_i) next_state = LOAD;
            end
            LOAD: begin
                if(input_counter == n - 1) next_state = PROCESSING;
            end
            PROCESSING: begin
                if(x_counter == n) next_state = STORE;
            end
            STORE: begin
                if(y_counter == m - 1) next_state = IDLE;
                else next_state = PROCESSING;
            end
            default: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // accumulator
    always_ff @(posedge clk_i) begin
        if(state == PROCESSING) begin
            if(x_counter == 0) begin
                accumulator <= product;
            end
            else begin
                accumulator <= accumulator + product;
            end
        end
        else if(state == STORE) begin
            accumulator <= 0;
        end
    end
    
    always_ff @(posedge clk_i) begin
        case(state)
            IDLE: begin
                x_counter <= 0;
                y_counter <= 0;
                input_counter <= 0; 
                accumulator <= 0;
            end
            LOAD: begin
                input_counter <= input_counter + 1;
                    
                if(next_state == PROCESSING) begin
                    x_counter <= 1;
                end
            end
            PROCESSING: begin
                x_counter <= x_counter + 1;
            end
            STORE: begin
                x_counter <= 0;
                y_counter <= y_counter + 1;
            end
        endcase
    end
    
    always_ff @(posedge clk_i or posedge rst_i) begin
        if(rst_i) begin
            state <= IDLE;
        end
        else begin
            state <= next_state;
        end
    end
    
endmodule
