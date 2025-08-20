`timescale 1ns / 1ps
import my_const::*;
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/16/2025 03:35:27 PM
// Design Name: 
// Module Name: accelerator_data_parser
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


module accelerator_data_parser(
    input logic clk_i,
    input logic rst_i,
    
    input logic [2:0] cmd_i,
    
    output logic en_o,
    
    input logic val_i,
    
    output logic [N-1:0] data_o,
    output logic [N-1:0] weight_o,
    output logic [N-1:0] bias_o,
    
    // BRAM Access
    // BRAM_0 - DATA
    // RW
    output logic [31:0]BRAM_PORTB_0_addr,
    output logic [31:0]BRAM_PORTB_0_din,
    input logic [31:0]BRAM_PORTB_0_dout,
    output [3:0]BRAM_PORTB_0_we,
    
    // BRAM_1 - WEIGHT
    // R
    output logic [31:0]BRAM_PORTB_1_addr,
    input logic [31:0]BRAM_PORTB_1_dout,
    
    // BRAM_2 - BIAS
    // R
    output logic [31:0]BRAM_PORTB_2_addr,
    input logic [31:0]BRAM_PORTB_2_dout
    );
    
    
    // resposible for receiving commands from the control unit
    // and sending in data in the correct manner for each computation unit
    // some complex state machines incoming :-)
    
    enum logic [2:0] {
        IDLE,
        CONV_0,
        CONV_1,
        POOL_0,
        POOL_1,
        DENSE_0,
        DENSE_1,
        DENSE_2,
        DONE
    } state, next_state;
    
    logic delay; // just a one cycle delay to accomodate for the BRAM delay
    logic [1:0] fetch_counter;
    logic [1:0] offset_counter;
    
    // data will be called feature interally to avoid confusion
    logic [31:0] rfeature_addr;
    logic [31:0] wfeature_addr;
    logic [31:0] rfeature_data;
    logic [31:0] wfeature_data;
    logic [3:0] feature_wea;
    
    logic [31:0] rweight_addr;
    logic [31:0] rweight_data;
    
    logic [31:0] rbias_addr;
    logic [31:0] rbias_data;
    
    // holds the cached word, to be broken up to 4 int8
    logic [31:0] cached_word_reg;
    
    assign BRAM_PORTB_0_dout = rfeature_data;
    assign BRAM_PORTB_0_din = wfeature_data;
    assign BRAM_PORTB_0_we = feature_wea;
    
    assign BRAM_PORTB_1_addr = rweight_addr;
    assign BRAM_PORTB_1_dout = rweight_data;
    
    assign BRAM_PORTB_2_addr = rbias_addr;
    assign BRAM_PORTB_2_dout = rbias_data;
    
    always_ff @(posedge clk_i) begin
        if(rst_i) begin
            delay <= 0;
            fetch_counter <= 0;
            offset_counter <= 0;
        end
        else begin
            case(state) 
                IDLE: begin
                    delay <= 0;
                    fetch_counter <= 0;
                    offset_counter <= 0;
                end
            endcase
        end
    end
    
    always_comb begin
        en_o = 0;
        data_o = 0;
        weight_o = 0;
        bias_o = 0;
    end
    
    always_comb begin
        next_state = state;
        
        case(cmd_i)
            CMD_IDLE: next_state = IDLE;
            CMD_CONVOLVE_0: next_state = CONV_0;
            CMD_CONVOLVE_1: next_state = CONV_1;
            CMD_POOL_0: next_state = POOL_0;
            CMD_POOL_1: next_state = POOL_1;
            CMD_DENSE_0: next_state = DENSE_0;
            CMD_DENSE_1: next_state = DENSE_1;
            CMD_DENSE_2: next_state = DENSE_2;
        endcase
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
