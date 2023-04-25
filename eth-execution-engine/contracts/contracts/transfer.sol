pragma solidity >=0.4.0 <0.7.0;

contract Transfer {
    mapping(string => uint256) account;

    function deposit(string memory arg0, uint256 arg1) public {
        account[arg0] = arg1;
    }

    function transfer(string memory arg0, string memory arg1, uint256 arg2) public {
        uint256 bal1 = account[arg0];
        uint256 bal2 = account[arg1];
        uint256 ammount = arg2;

        bal1 -= ammount;
        bal2 += ammount;

        account[arg0] = bal1;
        account[arg1] = bal2;
    }

    function get_balance(string memory arg0) public view returns (uint256 balance) {
        balance = account[arg0];
        return balance;
    }
}