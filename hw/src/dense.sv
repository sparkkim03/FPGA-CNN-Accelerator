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
    input logic signed [N-1:0] bias_i, // bias for the dense layer
    input logic signed [N-1:0] weight_i,
    
    output logic signed [N-1:0] data_o, // output data
    output logic val_dense_o, // valid signal for the output data
    output logic done_dense_o // done signal for the dense layer
);
    localparam BRAM_ADDR_WIDTH = 8;
    localparam ZEXT = BRAM_ADDR_WIDTH - $clog2(n);
    
    logic wea;
    // we just use the address width as needed
    // rest of the BRAM might be unused
    logic [$clog2(n)-1:0] data_addr_write;
    logic [$clog2(n)-1:0] data_addr_read; 
    
    logic signed [N-1:0] data_buffer;
    logic signed [N-1:0] bias_buffer;
    logic signed [N-1:0] weight_buffer;
    
    logic signed [N-1:0] bias_mem [m-1:0];
    
    logic [$clog2(m)-1:0] bias_counter;
    logic [$clog2(n)-1:0] data_counter;
    
    logic [$clog2(n):0] x_counter; // inner loop counter, multiplexed to input_addr
    logic [$clog2(m):0] y_counter; // outer loop counter
    
    logic signed [N-1:0] accumulator;
    logic signed [N-1:0] data_temp;
    logic signed [N*2-1:0] product;

    dense_bram data_mem(
        .addra({{(ZEXT){1'b0}}, data_addr_write}),
        .clka(clk_i),
        .dina(data_i),
        .wea(wea),
        .addrb({{(ZEXT){1'b0}}, data_addr_read}),
        .clkb(clk_i),
        .doutb(data_buffer)
    );
    
    enum logic [2:0] {
        IDLE,
        LOAD_BIAS,
        LOAD_DATA,
        PROCESSING,
        DONE
    } state, next_state;  
    
    assign wea = (state == LOAD_DATA);
    assign data_addr_write = data_counter;
    assign data_addr_read = x_counter;
    assign product = data_buffer * weight_i;
    assign done_dense_o = (state == DONE);
    
    always_comb begin
        data_o = 0;
        
        case(state) 
            PROCESSING: begin
                // applying bias and relu
                data_temp = accumulator + bias_mem[y_counter-1];
                data_o = (data_temp > 0) ? data_temp : 0;   
            end
        endcase
    end
    
    always_comb begin
        next_state = state;
        case(state)
            IDLE: begin
                if(en_i) next_state = LOAD_BIAS;
            end
            LOAD_BIAS: begin
                if(bias_counter == m-1) next_state = LOAD_DATA;
            end
            LOAD_DATA: begin
                if(data_counter == n-1) next_state = PROCESSING;
            end
            PROCESSING: begin
                if(y_counter == m) next_state = DONE;
            end
            DONE: begin
                next_state = IDLE;
            end
            default: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // accumulator
    always_ff @(posedge clk_i) begin
        if(state == PROCESSING) begin        
            accumulator <= accumulator + product; 
            
            if(x_counter == 0) begin
                accumulator <= 0;
            end
        end
    end
    
    always_ff @(posedge clk_i) begin
        bias_buffer <= bias_i;
        if(state == LOAD_BIAS) begin
            bias_mem[bias_counter] <= bias_i;
        end
    end
    
    always_ff @(posedge clk_i) begin
        weight_buffer <= weight_i;
    end
    
    always_ff @(posedge clk_i) begin
        if(rst_i) begin
            x_counter <= 0;
            y_counter <= 0;
            data_counter <= 0;
            bias_counter <= 0; 
            accumulator <= 0;
        end
        else begin
            case(state)
                IDLE: begin
                    x_counter <= 0;
                    y_counter <= 0;
                    data_counter <= 0;
                    bias_counter <= 0; 
                    accumulator <= 0;
                    val_dense_o <= 0;
                end
                LOAD_BIAS: begin
                    bias_counter <= bias_counter + 1;
                end
                LOAD_DATA: begin
                    data_counter <= data_counter + 1;
                        
                    if(next_state == PROCESSING) begin
                        x_counter <= 1;
                    end
                end
                PROCESSING: begin
                    if(x_counter == n) begin
                        x_counter <= 0;
                        
                        val_dense_o <= 1;
                        
                        y_counter <= y_counter + 1;
                    end
                    else begin
                        val_dense_o <= 0;
                        x_counter <= x_counter + 1;
                    end
                end
            endcase
        end
    end
    
    always_ff @(posedge clk_i) begin
        if(rst_i) begin
            state <= IDLE;
        end
        else begin
            state <= next_state;
        end
    end
    
endmodule