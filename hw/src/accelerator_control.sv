`timescale 1ns / 1ps
`include "const.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/15/2025 02:30:59 PM
// Design Name: 
// Module Name: accelerator_control
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


module accelerator_control(
    input logic clk_i,
    input logic rst_i,
    input logic done_i, // from the data router module
    
    output logic [1:0] cmd_o, // internal command signal
    
    input logic [31:0] cmd_reg_i,
    input logic [31:0] status_reg_i,
    
    output logic [31:0] cmd_reg_o,
    output logic cmd_valid,
    
    output logic [31:0] status_reg_o
    );
    
    enum logic [2:0] {
        IDLE,
        CONV,
        POOL,
        DENSE,
        ERR,
        DONE
    } state, next_state;
    
    // The driver should check if we're busy or not before they request
    // so I don't think we need any redundancy here?
    
    always_comb begin
        next_state = state;
        
        case(state) 
            IDLE: begin
                if(cmd_reg_i == `CMD_CONVOLVE) begin
                    next_state = CONV;
                end
                if(cmd_reg_i == `CMD_POOL) begin
                    next_state = POOL;
                end
                if(cmd_reg_i == `CMD_DENSE) begin
                    next_state = DENSE;
                end
            end
            CONV, POOL, DENSE: begin
                if(done_i) begin
                    next_state = DONE;
                end
            end
            DONE: begin
                next_state = IDLE;
            end
        endcase
    end
    
    always_comb begin
        // the cmd output will always be 0 so that right after we read the command
        // we clear the cmd register
        // cmd register cleared when cmd_valid = 1
        cmd_reg_o = 'b0;
        cmd_valid = 'b0;
        cmd_o = `CMD_IDLE;
        
        status_reg_o = 'b0;
        
        case(state)
            IDLE: begin
                status_reg_o = `STATUS_IDLE;
            end
            CONV: begin
                status_reg_o = `STATUS_BUSY;
                cmd_valid = 'b1;
                cmd_o = `CMD_CONVOLVE;
            end
            POOL: begin
                status_reg_o = `STATUS_BUSY;
                cmd_valid = 'b1;
                cmd_o = `CMD_POOL;
            end
            DENSE: begin
                status_reg_o = `STATUS_BUSY;
                cmd_valid = 'b1;
                cmd_o = `CMD_DENSE;
            end
            ERR: begin
                status_reg_o = `STATUS_ERR;
                cmd_valid = 'b1;
            end
            DONE: begin
                status_reg_o = `STATUS_DONE;
                cmd_valid = 'b1;
            end
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
